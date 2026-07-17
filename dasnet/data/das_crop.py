#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random-crop training dataset for large DAS segments.

Each sample is a ``crop_channels`` by ``crop_samples`` window read directly
from an HDF5 file.

Design:

* **Event-driven sampling**: the dataset is indexed by annotation. The window
  is placed so the anchor event keeps >= ``min_picks_anchor`` picks per phase
  inside the crop (retries + bbox-centred fallback).
* **Annotation clipping**: all instances of the image are clipped to the
  window — polylines (per mask stream) keep only in-window picks and are
  shifted to window coords; bbox / confident_area rects are intersected and
  shifted. Instances with fewer than ``min_picks_keep`` in-window picks on any
  active stream are dropped.
* **Single-channel input**: ``image_channels=1`` outputs the normalized
  strain-rate band only (no filtering -> no crop edge effects).
  ``image_channels=3`` reproduces the classic strain-rate/bandpass/highpass
  trio, filtering with a time-margin pad to suppress edge effects.
* **Noise stacking**: a same-size window from a random noise H5
  (``noise_csv``), gated by ``allowed_bins`` like the parent class.
* **Background windows** (interface reserved, default off):
  ``background_prob > 0`` makes some samples event-free windows with empty
  targets, for future negative-sample experiments.

Inherits COCO bookkeeping, mask-stream config and rasterizers from
``DASTrainDataset``; does not modify it.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torchvision.transforms import functional as F

from .das import (
    DASTrainDataset,
    _safe_design_sos_bandpass,
    _safe_design_sos_highpass,
    all_instances_in_allowed_bins,
    filter_instances_by_min_bin,
    normalize,
)
from scipy.signal import sosfiltfilt


def _clip_polyline(
    seg: Sequence[float],
    t0: int,
    t1: int,
    c0: int,
    c1: int,
) -> List[float]:
    """Keep [t,ch,...] points inside the window and shift to window coords."""
    out: List[float] = []
    for i in range(0, len(seg) - 1, 2):
        t = float(seg[i])
        ch = float(seg[i + 1])
        if t0 <= t < t1 and c0 <= ch < c1:
            out.extend([t - t0, ch - c0])
    return out


def _clip_rect(
    rect: Sequence[float],
    t0: int,
    t1: int,
    c0: int,
    c1: int,
) -> Optional[List[float]]:
    """Intersect [x_t, y_ch, w_t, h_ch] with the window; shift to window coords."""
    x, y, w, h = (float(v) for v in rect)
    nx0 = max(x, t0)
    ny0 = max(y, c0)
    nx1 = min(x + w, t1)
    ny1 = min(y + h, c1)
    if nx1 - nx0 < 1e-6 or ny1 - ny0 < 1e-6:
        return None
    return [nx0 - t0, ny0 - c0, nx1 - nx0, ny1 - ny0]


class DASCropTrainDataset(DASTrainDataset):
    """Random-window variant of :class:`DASTrainDataset` for huge segments."""

    def __init__(
        self,
        ann_path: str,
        root_dir: str,
        # window geometry (original sample units)
        crop_channels: int = 2048,
        crop_samples: int = 4096,
        resize_scale: float = 1.0,
        # anchor / keep thresholds (picks per phase inside the window)
        min_picks_anchor: int = 64,
        min_picks_keep: int = 8,
        max_window_tries: int = 10,
        # input construction
        image_channels: int = 1,
        filter_pad_samples: int = 256,
        # background-window interface (reserved; default off)
        background_prob: float = 0.0,
        max_background_tries: int = 10,
        # everything else forwarded to DASTrainDataset
        **kwargs: Any,
    ):
        kwargs.setdefault("resize_scale", resize_scale)
        # event stacking is not supported in crop mode (windows from different
        # files have unrelated geometry); force off regardless of caller.
        kwargs["enable_stack_event"] = False
        super().__init__(ann_path=ann_path, root_dir=root_dir, **kwargs)

        self.crop_channels = int(crop_channels)
        self.crop_samples = int(crop_samples)
        self.min_picks_anchor = int(min_picks_anchor)
        self.min_picks_keep = int(min_picks_keep)
        self.max_window_tries = int(max_window_tries)
        if image_channels not in (1, 3):
            raise ValueError(f"image_channels must be 1 or 3, got {image_channels}")
        self.image_channels = int(image_channels)
        self.filter_pad_samples = int(filter_pad_samples)
        self.background_prob = float(background_prob)
        self.max_background_tries = int(max_background_tries)

        # annotation-driven index
        self.ann_ids: List[int] = sorted(self.coco.anns.keys())
        # image dims from COCO (width = n_samples/time, height = n_channels)
        self._img_meta: Dict[int, Tuple[int, int]] = {
            int(img["id"]): (int(img["width"]), int(img["height"]))
            for img in self.coco.imgs.values()
        }
        self._noise_dims_cache: Dict[str, Tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self.ann_ids)

    # -----------------------------
    # raw window loading
    # -----------------------------
    def _read_window_raw(
        self, h5_path: str, c0: int, c1: int, t0: int, t1: int
    ) -> Tuple[np.ndarray, float, int, int]:
        """Read the window slab (with time pad for filtering).

        Returns (raw, dt, lo, hi) where raw[:, lo:hi] is the unpadded window.
        """
        pad = self.filter_pad_samples if self.image_channels == 3 else 0
        with h5py.File(h5_path, "r") as f:
            dset = f[self.data_key]
            n_t = int(dset.shape[1])
            dt = float(dset.attrs.get("dt_s", 0.01))
            tp0 = max(0, t0 - pad)
            tp1 = min(n_t, t1 + pad)
            raw = np.asarray(dset[c0:c1, tp0:tp1], dtype=np.float32)
        if not self.data_is_strain_rate:
            sr = np.diff(raw, axis=1) / dt
            raw = np.concatenate([np.zeros((sr.shape[0], 1), dtype=sr.dtype), sr], axis=1)
        return raw, dt, t0 - tp0, (t0 - tp0) + (t1 - t0)

    def _window_to_image(
        self, h5_path: str, c0: int, c1: int, t0: int, t1: int
    ) -> np.ndarray:
        """Build the (H=channels, W=time, C) float32 window image."""
        raw, dt, lo, hi = self._read_window_raw(h5_path, c0, c1, t0, t1)
        if self.image_channels == 1:
            ch0 = normalize(raw[:, lo:hi].T).T
            img = ch0[:, :, None].astype(np.float32)
        else:
            sos_bp = _safe_design_sos_bandpass(dt, self.f_band[0], self.f_band[1], order=4)
            sos_hp = _safe_design_sos_highpass(dt, self.f_high, order=4)
            sr_bp = sosfiltfilt(sos_bp, raw, axis=1)[:, lo:hi]
            sr_hp = sosfiltfilt(sos_hp, raw, axis=1)[:, lo:hi]
            sr = raw[:, lo:hi]
            img = np.zeros((sr.shape[0], sr.shape[1], 3), dtype=np.float32)
            img[:, :, 0] = normalize(sr.T).T
            img[:, :, 1] = normalize(sr_bp.T).T
            img[:, :, 2] = normalize(sr_hp.T).T
        return img

    # -----------------------------
    # window sampling
    # -----------------------------
    @staticmethod
    def _count_picks_in_window(
        seg: Sequence[float], t0: int, t1: int, c0: int, c1: int
    ) -> int:
        n = 0
        for i in range(0, len(seg) - 1, 2):
            if t0 <= seg[i] < t1 and c0 <= seg[i + 1] < c1:
                n += 1
        return n

    def _streams_for_ann(self, ann: Dict[str, Any]) -> List[List[float]]:
        """Active polylines (flat lists) for this annotation, one per stream."""
        cat = int(ann["category_id"])
        n_s = int(self._msc.get(cat, 1))
        out: List[List[float]] = []
        for k in range(n_s):
            raw = ann.get(self._stream_keys[k], [])
            seg = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else []
            out.append(list(seg))
        return out

    def _window_dims(self, img_w: int, img_h: int) -> Tuple[int, int]:
        """Effective crop size (time, channels), clamped to the image."""
        return min(self.crop_samples, img_w), min(self.crop_channels, img_h)

    def _sample_window_for_anchor(
        self, ann: Dict[str, Any], img_w: int, img_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Pick (t0, t1, c0, c1) so the anchor keeps enough picks per stream."""
        win_t, win_c = self._window_dims(img_w, img_h)
        streams = self._streams_for_ann(ann)
        if not streams or any(len(s) < 2 for s in streams):
            return None

        bx, by, bw, bh = (float(v) for v in ann["bbox"])  # x=time, y=channel
        # any t0/c0 in these ranges gives >=1 sample of window-bbox overlap;
        # the pick-count check below does the real gating
        t_lo = int(np.clip(bx - win_t + 1, 0, img_w - win_t))
        t_hi = int(np.clip(bx + bw - 1, t_lo, img_w - win_t))
        c_lo = int(np.clip(by - win_c + 1, 0, img_h - win_c))
        c_hi = int(np.clip(by + bh - 1, c_lo, img_h - win_c))

        # required picks: lower the bar for events smaller than the threshold
        n_min = [
            max(1, min(self.min_picks_anchor, (len(s) // 2) // 2)) for s in streams
        ]

        best: Optional[Tuple[int, Tuple[int, int, int, int]]] = None
        for _ in range(self.max_window_tries):
            t0 = random.randint(t_lo, t_hi)
            c0 = random.randint(c_lo, c_hi)
            t1, c1 = t0 + win_t, c0 + win_c
            counts = [self._count_picks_in_window(s, t0, t1, c0, c1) for s in streams]
            if all(c >= m for c, m in zip(counts, n_min)):
                return t0, t1, c0, c1
            score = min(counts)
            if best is None or score > best[0]:
                best = (score, (t0, t1, c0, c1))

        # fallback: centre the window on the bbox centre (clamped)
        tc = int(bx + bw / 2.0)
        cc = int(by + bh / 2.0)
        t0 = int(np.clip(tc - win_t // 2, 0, img_w - win_t))
        c0 = int(np.clip(cc - win_c // 2, 0, img_h - win_c))
        t1, c1 = t0 + win_t, c0 + win_c
        counts = [self._count_picks_in_window(s, t0, t1, c0, c1) for s in streams]
        if all(c >= m for c, m in zip(counts, n_min)):
            return t0, t1, c0, c1
        if best is not None and best[0] >= max(1, self.min_picks_keep):
            return best[1]
        return None

    def _sample_background_window(
        self, img_id: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Window with zero overlap with any event bbox of this image."""
        img_w, img_h = self._img_meta[img_id]
        win_t, win_c = self._window_dims(img_w, img_h)
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        for _ in range(self.max_background_tries):
            t0 = random.randint(0, max(0, img_w - win_t))
            c0 = random.randint(0, max(0, img_h - win_c))
            t1, c1 = t0 + win_t, c0 + win_c
            clear = True
            for ann in anns:
                bx, by, bw, bh = (float(v) for v in ann["bbox"])
                if not (bx + bw <= t0 or bx >= t1 or by + bh <= c0 or by >= c1):
                    clear = False
                    break
            if clear:
                return t0, t1, c0, c1
        return None

    # -----------------------------
    # annotation clipping
    # -----------------------------
    def _clip_annotations_to_window(
        self,
        annotations: List[Dict[str, Any]],
        t0: int,
        t1: int,
        c0: int,
        c1: int,
    ) -> List[Dict[str, Any]]:
        """Clip each instance; drop those with too few in-window picks."""
        out: List[Dict[str, Any]] = []
        for ann in annotations:
            cat = int(ann["category_id"])
            n_s = int(self._msc.get(cat, 1))

            clipped_streams: List[List[float]] = []
            ok = True
            for k in range(n_s):
                raw = ann.get(self._stream_keys[k], [])
                seg = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else []
                cs = _clip_polyline(seg, t0, t1, c0, c1)
                if len(cs) // 2 < self.min_picks_keep:
                    ok = False
                    break
                clipped_streams.append(cs)
            if not ok:
                continue

            rect = _clip_rect(ann["bbox"], t0, t1, c0, c1)
            if rect is None:
                continue

            ca_in = ann.get("confident_area") or ann.get("attention_mask") or []
            ca_out = []
            for r in ca_in:
                cr = _clip_rect(r, t0, t1, c0, c1)
                if cr is not None:
                    ca_out.append(cr)

            new_ann = dict(ann)
            new_ann["bbox"] = rect
            new_ann["area"] = float(rect[2] * rect[3])
            new_ann["confident_area"] = ca_out
            new_ann.pop("attention_mask", None)
            for k in range(n_s):
                new_ann[self._stream_keys[k]] = [clipped_streams[k]]
            for k in range(n_s, len(self._stream_keys)):
                new_ann.pop(self._stream_keys[k], None)
            out.append(new_ann)
        return out

    # -----------------------------
    # noise stacking on the window
    # -----------------------------
    def _noise_dims(self, path: str) -> Tuple[int, int]:
        dims = self._noise_dims_cache.get(path)
        if dims is None:
            with h5py.File(path, "r") as f:
                sh = f[self.data_key].shape
            dims = (int(sh[0]), int(sh[1]))  # (n_ch, n_t)
            self._noise_dims_cache[path] = dims
        return dims

    def _maybe_add_noise_window(
        self, img: np.ndarray, annotations: List[Dict[str, Any]]
    ) -> np.ndarray:
        if not self.synthetic_noise or self.noise_df is None:
            return img
        if not all_instances_in_allowed_bins(annotations, self.allowed_bins):
            return img
        if np.random.rand() >= self.syn_prob:
            return img

        win_c, win_t = img.shape[0], img.shape[1]
        for _ in range(5):
            idx = random.randint(0, len(self.noise_df) - 1)
            path = str(self.noise_df.iloc[idx]["File Path"])
            try:
                n_ch, n_t = self._noise_dims(path)
            except (OSError, KeyError):
                continue
            if n_ch < win_c or n_t < win_t:
                continue
            c0 = random.randint(0, n_ch - win_c)
            t0 = random.randint(0, n_t - win_t)
            try:
                noise_img = self._window_to_image(path, c0, c0 + win_c, t0, t0 + win_t)
            except (OSError, KeyError):
                continue
            factor = random.uniform(*self.syn_factor_range)
            mixed = img + noise_img * factor
            for c in range(mixed.shape[2]):
                mixed[:, :, c] = normalize(mixed[:, :, c])
            return mixed
        return img

    # -----------------------------
    # main getitem
    # -----------------------------
    def __getitem__(self, index: int):
        ann = self.coco.anns[self.ann_ids[index]]
        img_id = int(ann["image_id"])
        img_w, img_h = self._img_meta[img_id]

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        annotations = filter_instances_by_min_bin(annotations, self.min_keep_bins)
        anchor_alive = any(a["id"] == ann["id"] for a in annotations)

        window = None
        as_background = False
        if self.background_prob > 0 and random.random() < self.background_prob:
            window = self._sample_background_window(img_id)
            as_background = window is not None
        if window is None and anchor_alive:
            window = self._sample_window_for_anchor(ann, img_w, img_h)
        if window is None:
            return self.__getitem__(random.randint(0, len(self.ann_ids) - 1))

        t0, t1, c0, c1 = window
        if as_background:
            clipped = []
        else:
            clipped = self._clip_annotations_to_window(annotations, t0, t1, c0, c1)
            if not clipped:
                return self.__getitem__(random.randint(0, len(self.ann_ids) - 1))

        h5_path = self._imgid_to_h5_path(img_id)
        img = self._window_to_image(h5_path, c0, c1, t0, t1)  # (H=ch, W=t, C)
        img = self._maybe_add_noise_window(img, clipped)

        image = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 1, 0)  # (C, W=t, H=ch)
        orig_W, orig_H = image.shape[1], image.shape[2]
        new_W = max(1, int(orig_W * self.resize_scale))
        new_H = max(1, int(orig_H * self.resize_scale))
        if (new_W, new_H) != (orig_W, orig_H):
            image = F.resize(image, (new_W, new_H))

        scale_t = new_W / max(orig_W, 1)
        scale_ch = new_H / max(orig_H, 1)
        new_h, new_w = new_W, new_H  # mask grid: (time, channel)

        boxes: List[List[float]] = []
        labels: List[int] = []
        confident_area_masks: List[np.ndarray] = []
        areas: List[float] = []
        iscrowd: List[int] = []
        K = len(self._stream_keys)
        stream_rows: List[List[np.ndarray]] = [[] for _ in range(K)]

        for a in clipped:
            x0, y0, bw, bh = a["bbox"]  # x=time, y=channel (window coords)
            boxes.append([y0 * scale_ch, x0 * scale_t, (y0 + bh) * scale_ch, (x0 + bw) * scale_t])
            labels.append(int(a["category_id"]))
            areas.append((bw * scale_t) * (bh * scale_ch))
            iscrowd.append(int(a.get("iscrowd", 0)))

            cat = int(a["category_id"])
            n_s = int(self._msc.get(cat, 1))
            for k in range(K):
                segs = a.get(self._stream_keys[k], []) if k < n_s else []
                segs = segs if isinstance(segs, list) else []
                stream_rows[k].append(
                    self._gaussian_line_mask_u8(segs, new_h, new_w, scale_t, scale_ch)
                )

            rects = a.get("confident_area") or []
            if rects:
                att = self._rasterize_confident_area_rects(
                    rects, mask_shape=(new_h, new_w), original_shape=(orig_W, orig_H)
                )
            else:
                att = np.zeros((new_h, new_w), dtype=np.uint8)
            confident_area_masks.append(att)

        n_inst = len(labels)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(n_inst, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
        if n_inst == 0:
            z = torch.zeros(0, new_h, new_w, dtype=torch.uint8)
            mask_streams_t = tuple(z for _ in range(K))
            ca_t = torch.zeros(0, new_h, new_w, dtype=torch.uint8)
        else:
            mask_streams_t = tuple(
                torch.stack([torch.tensor(stream_rows[k][i], dtype=torch.uint8) for i in range(n_inst)])
                for k in range(K)
            )
            ca_t = torch.stack([torch.tensor(a, dtype=torch.uint8) for a in confident_area_masks])
        masks = mask_streams_t[0]
        areas_t = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.enable_vflip and n_inst > 0 and (random.random() < self.vflip_prob):
            # Mirror the cable direction. Tensor layout is (C, time, channel),
            # and boxes are [channel_min, time_min, channel_max, time_max].
            image = F.hflip(image)
            W_ch = image.shape[2]
            boxes_t[:, [0, 2]] = W_ch - boxes_t[:, [2, 0]]
            mask_streams_t = tuple(torch.flip(mask_streams_t[k], dims=[2]) for k in range(K))
            masks = mask_streams_t[0]
            ca_t = torch.flip(ca_t, dims=[2])

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks,
            "confident_area_masks": ca_t,
            "image_id": img_id,
            "area": areas_t,
            "iscrowd": iscrowd_t,
            "mask_streams": mask_streams_t,
            "crop_window": torch.as_tensor([t0, t1, c0, c1], dtype=torch.int64),
        }
        if K >= 2:
            target["masks_aux"] = mask_streams_t[1]
        return image, target
