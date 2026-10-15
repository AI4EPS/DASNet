"""
Learned temporal window for the mask branch (replaces rule-based ``crop_mask_proposals_and_expand``).

Predicts a time anchor inside each detection proposal, then builds
``[t_anchor - before_frac * L, t_anchor + after_frac * L]`` with
``L = max(1, alpha * (proposal_x2 - proposal_x1))``.

The interval is not clamped to the proposal or image. Keep the original coordinate
system for feature pooling, target projection, and mask pasting. RoIAlign handles
out-of-bounds samples; no image-wide shift or feature expansion is needed.
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

    Returns boxes in original image coordinates (possibly outside the image),
    unshifted anchors, and unchanged image shapes. The same boxes must be used
    for mask feature pooling, GT projection, and inference mask pasting.
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
    output_image_shapes: List[Tuple[int, int]] = []

    for i, prop in enumerate(proposals_per_image):
        if prop.numel() == 0:
            out_list.append(prop)
            output_image_shapes.append(image_shapes[i])
            continue
        logits_i = chunks[i]
        H_i, W_i = image_shapes[i]
        x1, y1, x2, y2 = prop.unbind(dim=1)
        L_t = (float(alpha) * (x2 - x1)).clamp(min=1.0)
        h = (y2 - y1).clamp(min=1.0)
        frac = torch.sigmoid(logits_i)
        y_anchor = y1 + frac * h
        y0p = y_anchor - float(before_frac) * L_t
        y1p = y_anchor + float(after_frac) * L_t
        ya, yb = torch.minimum(y0p, y1p), torch.maximum(y0p, y1p)

        # Keep even negative/out-of-image coordinates: shifting boxes alone
        # changes the sampled signal and couples otherwise independent proposals.
        new_boxes = torch.stack((x1, ya, x2, yb), dim=1)
        out_list.append(new_boxes)
        y_parts.append(y_anchor)
        output_image_shapes.append((H_i, W_i))

    if not y_parts:
        return out_list, anchor_logits.new_zeros((0,)), list(image_shapes)
    return out_list, torch.cat(y_parts, dim=0), output_image_shapes
