"""Resolve backbone settings before construction; old checkpoints remain ResNet50."""
from .backbones import SUPPORTED_BACKBONES


def saved_arguments(checkpoint):
    value = (checkpoint or {}).get("args") or {}
    return value if isinstance(value, dict) else vars(value)


def resolve_backbone(args, checkpoint=None):
    saved = saved_arguments(checkpoint)
    for field, default, choices in [
        ("backbone", "swin_t", SUPPORTED_BACKBONES),
        ("backbone_weights", "imagenet", ("imagenet", "none")),
    ]:
        requested = getattr(args, field, None)
        legacy_default = "resnet50" if field == "backbone" else default
        stored = saved.get(field) or legacy_default
        if checkpoint is not None and requested is not None and requested != stored:
            raise ValueError(f"Checkpoint {field}={stored!r}, requested {requested!r}. Changing architecture/initialization requires a new training run.")
        value = stored if checkpoint is not None else requested or default
        if value not in choices:
            raise ValueError(f"Unsupported {field}: {value!r}; choose from {choices}")
        setattr(args, field, value)
    return args


def restore_prediction_config(args, checkpoint):
    """Training checkpoint config is authoritative for model and input geometry."""
    resolve_backbone(args, checkpoint)
    saved = saved_arguments(checkpoint)
    if args.num_classes is not None and "num_classes" in saved and args.num_classes != saved["num_classes"]:
        raise ValueError("--num-classes conflicts with training checkpoint")
    fields = ("num_classes", "image_channels", "resize_scale", "transform_min_size",
              "transform_max_size", "mask_streams_by_category", "use_learned_temporal_crop",
              "loss_temporal_crop_weight", "temporal_crop_loss_divisor", "arrival_band_alpha",
              "arrival_band_before_frac", "arrival_band_after_frac", "rpn_anchor_sizes",
              "rpn_anchor_aspect_ratios", "rpn_anchor_auto", "rpn_anchor_scales")
    for field in fields:
        if field in saved:
            setattr(args, field, saved[field])
    if saved:
        args.mask_prepare_mode = saved.get("mask_prepare_mode", "fixed_crop")
        args.rpn_anchor_samples = saved.get("crop_samples", args.rpn_anchor_samples)
        args.rpn_anchor_channels = saved.get("crop_channels", args.rpn_anchor_channels)
    if args.num_classes is None:
        raise ValueError("Checkpoint has no num_classes metadata; provide --num-classes")
    return args
