"""
Learned temporal window for the mask branch (replaces rule-based ``crop_mask_proposals_and_expand``).

Predicts a time anchor inside each detection proposal, then builds
``[t_anchor - before_frac * L, t_anchor + after_frac * L]`` with
``L = max(1, alpha * W_image)`` (same as ``arrival_band``).

The interval is **not** clamped to the proposal: if it exceeds the current image time extent,
callers should zero-pad the FPN (``expand_features`` + larger ``image_shapes``), mirroring the
original crop pipeline's aspect-ratio-friendly expansion rather than shrinking the window.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

class TemporalWindowPredictor(nn.Module):
    """Maps pooled RoI features to one logit per RoI (sigmoid → position along proposal time)."""

    def __init__(self, in_channels: int = 256, spatial_size: int = 7, hidden: int = 512) -> None:
        super().__init__()
        flat = int(in_channels * spatial_size * spatial_size)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, roi_feat: torch.Tensor) -> torch.Tensor:
        # (N, C, S, S) -> (N,)
        return self.mlp(roi_feat).squeeze(-1)


def adjust_mask_proposals_temporal(
    proposals_per_image: List[torch.Tensor],
    anchor_logits: torch.Tensor,
    image_shapes: List[Tuple[int, int]],
    alpha: float,
    before_frac: float,
    after_frac: float,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[Tuple[int, int]]]:
    """
    Build mask RoI time windows from predicted anchors (no clamp to proposal time span).

    Returns adjusted boxes in the **expanded** logical grid, and ``expanded_image_shapes`` so
    callers can ``expand_features`` + ``mask_roi_pool`` like the rule-based crop path.

    If ``min(y_window) < 0``, applies a **global time shift** for that image so windows start at
    non-negative y (FPN pad is bottom-only in this codebase).
    """
    if not proposals_per_image:
        return [], anchor_logits.new_zeros((0,)), list(image_shapes)

    sizes = [int(p.shape[0]) for p in proposals_per_image]
    if anchor_logits.shape[0] != sum(sizes):
        raise ValueError(
            f"anchor_logits length {anchor_logits.shape[0]} != total proposals {sum(sizes)}"
        )
    chunks = torch.split(anchor_logits, sizes, dim=0)
    out_list: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    expanded_image_shapes: List[Tuple[int, int]] = []

    for i, prop in enumerate(proposals_per_image):
        if prop.numel() == 0:
            out_list.append(prop)
            expanded_image_shapes.append(image_shapes[i])
            continue
        logits_i = chunks[i]
        H_i, W_i = image_shapes[i]
        L = max(1.0, float(alpha) * float(W_i))
        L_t = prop.new_tensor(L)

        x1, y1, x2, y2 = prop.unbind(dim=1)
        h = (y2 - y1).clamp(min=1.0)
        frac = torch.sigmoid(logits_i)
        y_anchor = y1 + frac * h
        y0p = y_anchor - float(before_frac) * L_t
        y1p = y_anchor + float(after_frac) * L_t
        ya, yb = torch.minimum(y0p, y1p), torch.maximum(y0p, y1p)

        # Shift so min time >= 0 (bottom-only FPN padding cannot add virtual rows above 0).
        shift = torch.clamp(-torch.min(ya), min=torch.tensor(0.0, device=ya.device, dtype=ya.dtype))
        ya_s = ya + shift
        yb_s = yb + shift

        # Extend logical height with zeros at the bottom so the full window fits (like crop+expand).
        y_max_need = torch.max(yb_s)
        H_new = max(H_i, int(torch.ceil(y_max_need).item()) + 1)
        Hm1 = y1.new_tensor(float(H_new - 1))
        z = y1.new_tensor(0.0)
        y_lo = torch.clamp(ya_s, min=z, max=Hm1)
        y_hi = torch.clamp(yb_s, min=z, max=Hm1)
        y_hi = torch.maximum(y_hi, y_lo + 1.0)

        new_boxes = torch.stack((x1, y_lo, x2, y_hi), dim=1)
        out_list.append(new_boxes)
        y_parts.append(y_anchor)
        expanded_image_shapes.append((H_new, W_i))

    if not y_parts:
        return out_list, anchor_logits.new_zeros((0,)), list(image_shapes)
    return out_list, torch.cat(y_parts, dim=0), expanded_image_shapes
