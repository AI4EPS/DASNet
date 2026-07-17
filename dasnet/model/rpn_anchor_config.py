"""Derive RPN anchor sizes from training/inference input geometry."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple


def effective_transform_hw(
    height: int,
    width: int,
    *,
    resize_scale: float = 1.0,
    transform_min_size: int,
    transform_max_size: int,
) -> Tuple[int, int]:
    """Mirror ``GeneralizedRCNNTransform`` sizing on (H, W) = (time, channel)."""
    h = max(1, int(round(height * float(resize_scale))))
    w = max(1, int(round(width * float(resize_scale))))
    scale = float(transform_min_size) / float(min(h, w))
    if max(h, w) * scale > transform_max_size:
        scale = float(transform_max_size) / float(max(h, w))
    return max(1, int(h * scale + 0.5)), max(1, int(w * scale + 0.5))


def parse_anchor_scales(blob: str, num_levels: int = 5) -> Tuple[float, ...]:
    s = (blob or "").strip()
    if not s:
        scales = (0.125, 0.25, 0.5, 1.0, 1.0)
    else:
        scales = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    if not scales or any(v <= 0 for v in scales):
        raise ValueError("anchor scales must be positive")
    if len(scales) == 1:
        scales = scales * num_levels
    if len(scales) != num_levels:
        raise ValueError(
            f"expected {num_levels} anchor scale values (one per FPN level), got {len(scales)}"
        )
    return scales


def anchor_sizes_from_short_side(
    short_side: int,
    scales: Sequence[float],
    *,
    num_levels: int = 5,
) -> Tuple[Tuple[int, ...], ...]:
    if len(scales) != num_levels:
        raise ValueError(f"scales length {len(scales)} != num_levels {num_levels}")
    sizes = []
    prev = 0
    for scale in scales:
        size = max(1, int(round(short_side * float(scale))))
        size = max(size, prev)
        sizes.append(size)
        prev = size
    return tuple((size,) for size in sizes)


def compute_rpn_anchor_sizes(
    *,
    input_height: int,
    input_width: int,
    resize_scale: float,
    transform_min_size: int,
    transform_max_size: int,
    scales: Sequence[float],
    num_levels: int = 5,
) -> Tuple[Tuple[Tuple[int, ...], ...], dict]:
    eff_h, eff_w = effective_transform_hw(
        input_height,
        input_width,
        resize_scale=resize_scale,
        transform_min_size=transform_min_size,
        transform_max_size=transform_max_size,
    )
    short_side = min(eff_h, eff_w)
    anchor_sizes = anchor_sizes_from_short_side(short_side, scales, num_levels=num_levels)
    meta = {
        "input_height": int(input_height),
        "input_width": int(input_width),
        "effective_height": eff_h,
        "effective_width": eff_w,
        "short_side": short_side,
        "scales": tuple(float(s) for s in scales),
        "anchor_sizes": tuple(s[0] for s in anchor_sizes),
    }
    return anchor_sizes, meta


def resolve_rpn_anchor_sizes(
    explicit_sizes: Optional[Tuple[Tuple[int, ...], ...]],
    *,
    auto: bool,
    scales_blob: str,
    input_height: Optional[int],
    input_width: Optional[int],
    resize_scale: float,
    transform_min_size: int,
    transform_max_size: int,
    num_levels: int = 5,
) -> Tuple[Optional[Tuple[Tuple[int, ...], ...]], Optional[dict]]:
    if explicit_sizes is not None:
        return explicit_sizes, None
    if not auto:
        return None, None
    if input_height is None or input_width is None:
        raise ValueError("input_height and input_width are required when rpn_anchor_auto is enabled")
    scales = parse_anchor_scales(scales_blob, num_levels=num_levels)
    sizes, meta = compute_rpn_anchor_sizes(
        input_height=input_height,
        input_width=input_width,
        resize_scale=resize_scale,
        transform_min_size=transform_min_size,
        transform_max_size=transform_max_size,
        scales=scales,
        num_levels=num_levels,
    )
    return sizes, meta
