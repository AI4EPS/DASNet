"""
Lightweight **confident-area** gating on FPN features before mask RoIAlign.

Replaces or complements fixed geometric crop: multiply each feature level by a
per-image soft mask. Sources:

- **attention_gate**: max over GT ``confident_area_masks`` (training) or identity (eval).
  Optional **mask constraint**: multiply that map by a union of GT instance ``masks`` (optionally
  dilated) so the gate only applies where the confident-area map overlaps credible arrival geometry.
- **arrival_band**: per-instance time band from GT ``masks`` (mass center along **time rows** on
  **mask ∩ confident_area** when per-instance confident-area rasters exist, so placeholder
  arrivals outside the trusted region do not shift the centroid); channel extent is **only
  within that instance's bbox** on the mask grid (``x`` = channel,
  ``y`` = time, same as ``das.py``); time span ``L = alpha * W`` with ``W`` = full mask width
  in channel bins; asymmetric ``[t_mean - before_frac*L, t_mean + after_frac*L]``, union over
  instances, then floor; eval = identity.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def confident_area_stack_from_target(target: dict) -> Optional[torch.Tensor]:
    """
    Stacked per-instance raster (N, H, W). Prefers ``confident_area_masks``;
    falls back to legacy ``attention_masks``.
    """
    if "confident_area_masks" in target and target["confident_area_masks"] is not None:
        x = target["confident_area_masks"]
        if x.numel() > 0:
            return x
    if "attention_masks" in target and target["attention_masks"] is not None:
        x = target["attention_masks"]
        if x.numel() > 0:
            return x
    return None


def _dilate_binary_hw(mask_hw: torch.Tensor, radius_px: int) -> torch.Tensor:
    if radius_px <= 0:
        return mask_hw
    k = 2 * radius_px + 1
    x = mask_hw.unsqueeze(0).unsqueeze(0)
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=radius_px)[0, 0]


def _union_gt_masks_hw(
    target: dict,
    H: int,
    W: int,
    dev: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-pixel max over instances, values in ~[0, 1], shape (H, W)."""
    if "masks" not in target or target["masks"].numel() == 0:
        return torch.zeros(H, W, device=dev, dtype=dtype)
    masks = target["masks"].float() / 255.0
    if masks.ndim != 3 or masks.shape[0] == 0:
        return torch.zeros(H, W, device=dev, dtype=dtype)
    M = masks.max(dim=0).values.clamp(0.0, 1.0)
    if M.shape[-2:] != (H, W):
        M = F.interpolate(
            M.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return M


def aggregate_image_attention_gate(
    targets: List[dict],
    image_shapes: List[Tuple[int, int]],
    floor: float = 0.05,
    device: Optional[torch.device] = None,
    constrain_with_gt_masks: bool = False,
    mask_dilate_px: int = 0,
) -> List[torch.Tensor]:
    """
    Build one (H, W) gate per image in [floor, 1].

    Uses max over instance ``confident_area_masks`` then bilinear-resize to image size.
    If none or empty, returns all-ones (no gating).

    If ``constrain_with_gt_masks`` is True, the confident-area map is multiplied (before the
    floor clamp) by the union of GT instance ``masks`` (optional morphological dilation).
    Pixels with near-zero weight therefore do not open a gate; confident area far from any
    GT arrival stroke is suppressed. If there is no GT mask mass, falls back to all-ones
    so training does not collapse when ``masks`` are missing.
    """
    gates: List[torch.Tensor] = []
    for t, (H, W) in zip(targets, image_shapes):
        dev = device
        if dev is None:
            dev = t["boxes"].device if "boxes" in t and t["boxes"].numel() else torch.device("cpu")

        ca = confident_area_stack_from_target(t)
        if ca is None or ca.numel() == 0:
            gates.append(torch.ones(H, W, device=dev, dtype=torch.float32))
            continue

        am = ca.float()
        if am.ndim != 3:
            gates.append(torch.ones(H, W, device=dev, dtype=torch.float32))
            continue

        g = am.max(dim=0).values.clamp(0.0, 1.0)
        if g.shape[-2:] != (H, W):
            g = F.interpolate(
                g.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        if constrain_with_gt_masks:
            M = _union_gt_masks_hw(t, H, W, dev, g.dtype)
            M = _dilate_binary_hw(M, int(mask_dilate_px))
            if M.sum() < 1e-8:
                gates.append(torch.ones(H, W, device=dev, dtype=torch.float32))
                continue
            g = (g * M).clamp(0.0, 1.0)

        fl = torch.tensor(floor, device=g.device, dtype=g.dtype)
        g = torch.maximum(g, fl)
        gates.append(g)

    return gates


def _instance_confident_area_hw(target: dict, j: int, H: int, W: int, dev: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Per-instance confident-area map in ``[0, 1]``, shape ``(H, W)``, or ``None`` if unavailable."""
    ca = confident_area_stack_from_target(target)
    if ca is None or j < 0 or j >= ca.shape[0]:
        return None
    x = ca[j].float()
    if x.numel() == 0:
        return None
    if x.max() > 1.0 + 1e-6:
        x = x / 255.0
    x = x.clamp(0.0, 1.0)
    if x.shape[-2:] != (H, W):
        x = F.interpolate(x.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return x.to(device=dev, dtype=dtype)


def _arrival_band_region_for_instance_j(
    target: dict,
    j: int,
    H: int,
    W: int,
    alpha: float,
    before_frac: float,
    after_frac: float,
    class_ids: Tuple[int, ...],
    use_confident_area_for_centroid: bool = True,
    instance_masks: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, float | int]]:
    """Geometry for instance index ``j`` only; ``None`` if that instance does not contribute a band.

    Time centroid uses ``mask * confident_area`` (per-instance ``confident_area_masks``) when
    ``use_confident_area_for_centroid`` is True and that raster exists; if the intersection has
    negligible mass, returns ``None``. When no confident-area stack exists in ``target``, falls
    back to mask-only mass (legacy).
    """
    if before_frac <= 0 or after_frac <= 0:
        raise ValueError("before_frac and after_frac must be positive")
    allow = set(int(x) for x in class_ids) if class_ids else None
    if instance_masks is not None:
        dev = instance_masks.device
        masks = instance_masks.float()
        if masks.max() > 1.0 + 1e-6:
            masks = masks / 255.0
    else:
        dev = target["masks"].device
        masks = target["masks"].float() / 255.0
    labels = target["labels"]
    boxes = target.get("boxes")
    L = max(1.0, float(alpha) * float(W))
    if j < 0 or j >= masks.shape[0]:
        return None
    lab = int(labels[j].item())
    if allow is not None and lab not in allow:
        return None
    m = masks[j]
    if m.shape[-2:] != (H, W):
        m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    if m.sum() < 1e-6:
        return None

    m_for_centroid = m
    if use_confident_area_for_centroid:
        ca_j = _instance_confident_area_hw(target, j, H, W, dev, m.dtype)
        if ca_j is not None:
            m_ca = m * ca_j
            if m_ca.sum() < 1e-6:
                return None
            m_for_centroid = m_ca

    row_mass = m_for_centroid.sum(dim=1)
    if row_mass.sum() < 1e-6:
        return None
    r = torch.arange(H, device=dev, dtype=m.dtype)
    t_mean = float(((row_mass * r).sum() / row_mass.sum()).item())
    t_low = t_mean - float(before_frac) * L
    t_high = t_mean + float(after_frac) * L
    t0 = int(torch.floor(torch.tensor(t_low, device=dev)).item())
    t1 = int(torch.ceil(torch.tensor(t_high, device=dev)).item())
    t0 = max(0, min(H - 1, t0))
    t1 = max(0, min(H - 1, t1))
    if t1 < t0:
        t0, t1 = t1, t0
    c0, c1_excl = 0, W
    if boxes is not None and boxes.numel() > 0 and j < boxes.shape[0]:
        xc1, _yt1, xc2, _yt2 = boxes[j].detach().tolist()
        lo = float(min(xc1, xc2))
        hi = float(max(xc1, xc2))
        c0 = max(0, min(W, int(math.floor(lo))))
        c1_excl = max(0, min(W, int(math.ceil(hi))))
        if c1_excl <= c0:
            c1_excl = min(W, c0 + 1)
    return {"label": lab, "t_mean": t_mean, "t0": t0, "t1": t1, "c0": c0, "c1_excl": c1_excl, "L": float(L)}


def arrival_band_region_from_soft_mask(
    mask_hw: torch.Tensor,
    box_xyxy: torch.Tensor,
    label: int,
    image_hw: Tuple[int, int],
    alpha: float = 1.0,
    before_frac: float = 0.25,
    after_frac: float = 0.75,
    class_ids: Tuple[int, ...] = (),
) -> Optional[Dict[str, float | int]]:
    """
    Same time-band geometry as ``_arrival_band_region_for_instance_j``, for **inference**:
    time centroid from **predicted** soft ``mask_hw`` (no GT confident-area — not available at
    test time). ``mask_hw`` is ``(H, W)`` = (time, channel); ``box_xyxy`` is ``[x1,y1,x2,y2]``
    = (channel, time). Channel span ``L = max(1, alpha * W_image)`` matches training.
    """
    if before_frac <= 0 or after_frac <= 0:
        raise ValueError("before_frac and after_frac must be positive")
    allow = set(int(x) for x in class_ids) if class_ids else None
    if allow is not None and int(label) not in allow:
        return None

    H, W = int(image_hw[0]), int(image_hw[1])
    dev = mask_hw.device
    dtype = torch.float32
    m = mask_hw.detach().to(device=dev, dtype=dtype).clamp(0.0, 1.0)
    if m.ndim != 2:
        return None
    if m.shape[-2:] != (H, W):
        m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0, 0]

    if m.sum() < 1e-6:
        return None

    L = max(1.0, float(alpha) * float(W))
    row_mass = m.sum(dim=1)
    if row_mass.sum() < 1e-6:
        return None
    r = torch.arange(H, device=dev, dtype=dtype)
    t_mean = float(((row_mass * r).sum() / row_mass.sum()).item())
    t_low = t_mean - float(before_frac) * L
    t_high = t_mean + float(after_frac) * L
    t0 = int(torch.floor(torch.tensor(t_low, device=dev)).item())
    t1 = int(torch.ceil(torch.tensor(t_high, device=dev)).item())
    t0 = max(0, min(H - 1, t0))
    t1 = max(0, min(H - 1, t1))
    if t1 < t0:
        t0, t1 = t1, t0

    c0, c1_excl = 0, W
    if box_xyxy is not None and box_xyxy.numel() >= 4:
        xc1, _yt1, xc2, _yt2 = box_xyxy.detach().tolist()[:4]
        lo = float(min(xc1, xc2))
        hi = float(max(xc1, xc2))
        c0 = max(0, min(W, int(math.floor(lo))))
        c1_excl = max(0, min(W, int(math.ceil(hi))))
        if c1_excl <= c0:
            c1_excl = min(W, c0 + 1)

    return {"label": int(label), "t_mean": t_mean, "t0": t0, "t1": t1, "c0": c0, "c1_excl": c1_excl, "L": float(L)}


def single_instance_arrival_gate_map(
    target: dict,
    j: int,
    image_hw: Tuple[int, int],
    alpha: float = 1.0,
    before_frac: float = 0.25,
    after_frac: float = 0.75,
    class_ids: Tuple[int, ...] = (),
    use_confident_area_for_centroid: bool = True,
    instance_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Binary ``(H, W)`` gate for **one** GT instance (same geometry as ``arrival_band`` union logic,
    but no union). Used as teacher for per-RoI learned gate distillation.
    """
    H, W = int(image_hw[0]), int(image_hw[1])
    dev = instance_masks.device if instance_masks is not None else target["masks"].device
    G = torch.zeros(H, W, device=dev, dtype=torch.float32)
    reg = _arrival_band_region_for_instance_j(
        target,
        j,
        H,
        W,
        alpha,
        before_frac,
        after_frac,
        class_ids,
        use_confident_area_for_centroid=use_confident_area_for_centroid,
        instance_masks=instance_masks,
    )
    if reg is None:
        return G
    t0, t1 = int(reg["t0"]), int(reg["t1"])
    c0, c1_excl = int(reg["c0"]), int(reg["c1_excl"])
    G[t0 : t1 + 1, c0:c1_excl] = 1.0
    return G


def arrival_band_instance_regions(
    target: dict,
    image_hw: Tuple[int, int],
    alpha: float = 1.0,
    before_frac: float = 0.25,
    after_frac: float = 0.75,
    class_ids: Tuple[int, ...] = (),
    use_confident_area_for_centroid: bool = True,
    instance_masks_override: Optional[torch.Tensor] = None,
) -> List[Dict[str, float | int]]:
    """
    Same per-instance geometry as ``build_arrival_band_gates_from_gt_masks`` (for QA plots).

    Returns one dict per instance that contributes a band, with keys:
    ``label``, ``t_mean`` (float row index on time axis), ``t0``, ``t1`` (inclusive row indices),
    ``c0``, ``c1_excl`` (half-open channel column indices), ``L`` (float span used for time band).
    """
    if before_frac <= 0 or after_frac <= 0:
        raise ValueError("before_frac and after_frac must be positive")

    H, W = int(image_hw[0]), int(image_hw[1])
    out: List[Dict[str, float | int]] = []

    n_inst = int(instance_masks_override.shape[0]) if instance_masks_override is not None else int(target["masks"].shape[0])
    for j in range(n_inst):
        reg = _arrival_band_region_for_instance_j(
            target,
            j,
            H,
            W,
            alpha,
            before_frac,
            after_frac,
            class_ids,
            use_confident_area_for_centroid=use_confident_area_for_centroid,
            instance_masks=instance_masks_override,
        )
        if reg is not None:
            out.append(reg)

    return out


def build_arrival_band_gates_from_gt_masks(
    targets: List[dict],
    image_shapes: List[Tuple[int, int]],
    alpha: float = 1.0,
    before_frac: float = 0.25,
    after_frac: float = 0.75,
    class_ids: Tuple[int, ...] = (),
    floor: float = 0.05,
    use_confident_area_for_centroid: bool = True,
    gt_masks_override_per_image: Optional[List[Optional[torch.Tensor]]] = None,
) -> List[torch.Tensor]:
    """
    One (H, W) gate per image: union of horizontal (time) bands per qualifying instance.

    ``image_shapes`` and masks use **(H, W) = (time, channel)** on the mask grid — same as
    ``DASTrainDataset``: first index is **time**, second is **channel** (see ``das.py``:
    ``new_h, new_w = new_W, new_H`` and box convention ``x`` = channel, ``y`` = time).

    For each instance, mean **time** index ``t_mean`` uses mass on **``mask ∩ confident_area``**
    when per-instance confident-area rasters exist; otherwise mass on ``mask`` only. If confident
    area exists but has **no overlap** with that instance mask, the instance is skipped (no band).

    Time span ``L = max(1.0, float(alpha) * float(W))`` uses **full** mask width ``W`` in
    channel bins (not bbox width). Band **rows** (inclusive after clamping): from
    ``t_mean - before_frac * L`` to ``t_mean + after_frac * L``.

    Band **columns** are limited to that instance's bbox **channel** span (``x1``, ``x2`` from
    ``target['boxes'][j]``; ``[x1,y1,x2,y2]`` = channel/time in mask index space). Columns filled
    are ``floor(min(x1,x2)) : ceil(max(x1,x2))`` clamped to ``[0, W)`` (degenerate-safe). If boxes
    are missing for that index, falls back to all columns for that instance.

    - ``class_ids``: if non-empty, only instances with ``labels[j] in class_ids`` contribute bands.
    - Instances with near-zero mask mass are skipped.
    - If no band is drawn, gate is all-ones (safe fallback).
    """
    if before_frac <= 0 or after_frac <= 0:
        raise ValueError("before_frac and after_frac must be positive")

    gates: List[torch.Tensor] = []

    if gt_masks_override_per_image is None:
        gt_masks_override_per_image = [None] * len(targets)
    elif len(gt_masks_override_per_image) != len(targets):
        raise ValueError("gt_masks_override_per_image must match targets length")

    for t, (H, W), m_over in zip(targets, image_shapes, gt_masks_override_per_image):
        dev = t["masks"].device
        G = torch.zeros(H, W, device=dev, dtype=torch.float32)
        regions = arrival_band_instance_regions(
            t,
            (H, W),
            alpha,
            before_frac,
            after_frac,
            class_ids,
            use_confident_area_for_centroid=use_confident_area_for_centroid,
            instance_masks_override=m_over,
        )
        for reg in regions:
            t0, t1 = int(reg["t0"]), int(reg["t1"])
            c0, c1_excl = int(reg["c0"]), int(reg["c1_excl"])
            G[t0 : t1 + 1, c0:c1_excl] = torch.maximum(
                G[t0 : t1 + 1, c0:c1_excl],
                torch.ones_like(G[t0 : t1 + 1, c0:c1_excl]),
            )

        if G.sum() < 1e-6:
            gates.append(torch.ones(H, W, device=dev, dtype=torch.float32))
        else:
            fl = torch.tensor(floor, device=dev, dtype=G.dtype)
            gates.append(torch.maximum(G, fl))

    return gates


def apply_feature_attention_gate(
    features: Dict[str, torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    gates_hw: List[torch.Tensor],
) -> OrderedDict:
    """
    Multiply each FPN level by a per-image gate resized to that level's (H, W).

    Args:
        features: backbone FPN outputs, each (B, C, Hf, Wf)
        image_shapes: list of (H_img, W_img) per batch image (same order as gates_hw)
        gates_hw: list length B of (H_img, W_img) tensors
    """
    out = OrderedDict()
    B = next(iter(features.values())).shape[0]
    if B != len(gates_hw):
        raise ValueError(f"gates_hw length {len(gates_hw)} != batch {B}")

    for name, feat in features.items():
        pieces = []
        for b in range(B):
            g = gates_hw[b]
            if g.device != feat.device:
                g = g.to(feat.device)
            Hf, Wf = feat.shape[2], feat.shape[3]
            gb = F.interpolate(
                g.unsqueeze(0).unsqueeze(0),
                size=(Hf, Wf),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            gb = gb.unsqueeze(0).unsqueeze(0)
            pieces.append(feat[b : b + 1] * gb)
        out[name] = torch.cat(pieces, dim=0)
    return out
