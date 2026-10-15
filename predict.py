"""Run local DASNet inference on HDF5 files listed in a CSV."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from dasnet.data.das import DASInferDataset
from dasnet.model.dasnet import build_model as build_dasnet_model
from dasnet.model.backbones import SUPPORTED_BACKBONES
from dasnet.model.checkpoint_config import resolve_backbone, restore_prediction_config
from dasnet.model.rpn_anchor_config import resolve_rpn_anchor_sizes


DEFAULT_PLOT_ARRIVAL_CLASSES = (2, 3, 8)  # Monterey: Blue B, Fin, earthquake P/S.


def parse_anchor_sizes(value: str):
    """Parse comma-separated RPN anchor sizes, optionally grouped by semicolon."""
    groups = [group.strip() for group in value.split(";") if group.strip()]
    if not groups:
        return None
    parsed = [
        tuple(int(item.strip()) for item in group.split(",") if item.strip())
        for group in groups
    ]
    if any(not group or any(item <= 0 for item in group) for group in parsed):
        raise ValueError("RPN anchor sizes must be positive integers.")
    return tuple((item,) for item in parsed[0]) if len(parsed) == 1 else tuple(parsed)


def parse_anchor_ratios(value: str):
    """Parse semicolon-separated groups of positive RPN aspect ratios."""
    groups = [group.strip() for group in value.split(";") if group.strip()]
    if not groups:
        return None
    parsed = tuple(
        tuple(float(item.strip()) for item in group.split(",") if item.strip())
        for group in groups
    )
    if any(not group or any(item <= 0 for item in group) for group in parsed):
        raise ValueError("RPN aspect ratios must be positive.")
    return parsed


def parse_mask_streams(value: str) -> dict[int, int]:
    """Parse category-to-mask-stream mappings such as ``1:2,2:1``."""
    streams = {}
    for item in (part.strip() for part in value.split(",")):
        if not item:
            continue
        try:
            category, count = item.split(":", 1)
            streams[int(category)] = int(count)
        except ValueError as error:
            raise ValueError(
                f"Invalid mask stream mapping {item!r}; use category:count."
            ) from error
    if any(count < 1 for count in streams.values()):
        raise ValueError("Mask-stream counts must be positive.")
    return streams


def collate_batch(batch):
    return tuple(zip(*batch))


def read_input_paths(csv_path: str, from_raw: bool) -> tuple[list[str], list[str]]:
    """Read HDF5 paths from a standard or raw-input CSV."""
    frame = pd.read_csv(csv_path)
    path_column = "File Path" if from_raw and "File Path" in frame.columns else "path"
    if path_column not in frame.columns:
        raise ValueError(f"{csv_path} must contain a '{path_column}' column.")
    paths = frame[path_column].astype(str).tolist()
    names = (
        frame["File Name"].astype(str).tolist()
        if from_raw and "File Name" in frame.columns
        else [os.path.basename(path) for path in paths]
    )
    missing = [path for path in paths if not path.startswith("gs://") and not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Input file does not exist: {missing[0]}")
    return paths, names


def extract_picks(mask: np.ndarray, box: np.ndarray, threshold: float) -> tuple[list[list[float]], list[float]]:
    """Extract the strongest mask peak for every channel inside a detection box."""
    smoothed = gaussian_filter(mask, sigma=5)
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    x0, x1 = max(x0, 0), min(x1, smoothed.shape[1])
    y0, y1 = max(y0, 0), min(y1, smoothed.shape[0])
    if x0 >= x1 or y0 >= y1:
        return [], []
    points, scores = [], []
    for channel in range(x0, x1):
        column = smoothed[y0:y1, channel]
        if not column.size:
            continue
        offset = int(column.argmax())
        score = float(column[offset])
        if score >= threshold:
            points.append([float(channel), float(y0 + offset)])
            scores.append(score)
    return points, scores


def active_mask_streams(prediction: dict, index: int, streams_by_category: dict[int, int]):
    """Return the mask streams configured for a predicted instance."""
    count = streams_by_category.get(int(prediction["labels"][index]), 1)
    all_streams = prediction.get(
        "mask_streams_all",
        prediction.get("mask_streams", [prediction["masks"]]),
    )
    return [
        np.asarray(np.squeeze(stream[index]), dtype=np.float32)
        for stream in all_streams[:count]
    ]


def prediction_record(file_name: str, prediction: dict, args) -> dict:
    """Convert one model prediction to the compact public JSON format."""
    scale_back = 1.0 / args.resize_scale
    keep = np.flatnonzero(prediction["scores"] >= args.min_probability)
    instances = []
    for index in keep:
        box = prediction["boxes"][index]
        streams = active_mask_streams(prediction, index, args.mask_streams_by_category)
        stream_records = []
        for stream_index, mask in enumerate(streams):
            picks, pick_scores = extract_picks(mask, box, args.mask_threshold)
            stream_records.append(
                {
                    "stream": stream_index,
                    "picks": [[x * scale_back, y * scale_back] for x, y in picks],
                    "pick_scores": pick_scores,
                }
            )
        instances.append(
            {
                "box": [float(value) * scale_back for value in box],
                "score": float(prediction["scores"][index]),
                "label": int(prediction["labels"][index]),
                "picks_by_stream": stream_records,
            }
        )
    return {
        "file_name": os.path.basename(file_name),
        "resize_scale": args.resize_scale,
        "instances": instances,
    }


def keep_labels_inside_axes(figure, labels, margin_pixels=4):
    """Fit the entire text background inside its axes after layout is finalized."""
    renderer = figure.canvas.get_renderer()
    for text in labels:
        axis = text.axes
        bounds = axis.get_window_extent(renderer)
        text.update_bbox_position_size(renderer)
        box = text.get_bbox_patch().get_window_extent(renderer)
        while (box.width > bounds.width - 2 * margin_pixels or
               box.height > bounds.height - 2 * margin_pixels) and text.get_fontsize() > 1:
            text.set_fontsize(text.get_fontsize() * 0.9)
            text.update_bbox_position_size(renderer)
            box = text.get_bbox_patch().get_window_extent(renderer)
        dx = max(bounds.x0 + margin_pixels - box.x0,
                 min(0, bounds.x1 - margin_pixels - box.x1))
        dy = max(bounds.y0 + margin_pixels - box.y0,
                 min(0, bounds.y1 - margin_pixels - box.y1))
        position = text.get_transform().transform(text.get_position())
        text.set_position(text.get_transform().inverted().transform(position + [dx, dy]))


def save_prediction_figure(file_name: str, image: torch.Tensor, prediction: dict, args):
    """Show all detection boxes and arrival overlays for selected categories."""
    scores = prediction["scores"]
    keep = np.flatnonzero(scores >= args.min_probability)
    views = [image[min(index, image.shape[0] - 1)].numpy() for index in range(3)]
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, dpi=150)
    colors = ("cyan", "magenta", "lime")
    box_labels = []
    arrival_classes = set(getattr(args, "plot_arrival_classes", DEFAULT_PLOT_ARRIVAL_CLASSES))

    for panel_index, axis in enumerate(axes):
        background = views[panel_index].T
        low, high = np.percentile(background, (2, 98))
        axis.imshow(
            background,
            cmap="seismic",
            origin="lower",
            aspect="auto",
            vmin=low,
            vmax=high,
            rasterized=True,
        )
        for index in keep:
            label = int(prediction["labels"][index])
            channel_min, time_min, channel_max, time_max = (
                float(value) for value in prediction["boxes"][index]
            )
            axis.add_patch(
                Rectangle(
                    (time_min, channel_min),
                    time_max - time_min,
                    channel_max - channel_min,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
            )
            text = axis.text(
                time_min + 20,
                channel_max - 10,
                f"Class {label}\n{scores[index]:.2f}",
                color="black",
                fontsize=8,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
                clip_on=True,
                zorder=10,
            )
            text.set_in_layout(False)
            box_labels.append(text)
            if label not in arrival_classes:
                continue
            for stream_index, mask in enumerate(
                active_mask_streams(prediction, index, args.mask_streams_by_category)
            ):
                display_mask = np.clip(mask.T, 0.0, 1.0)
                rgba = plt.get_cmap(("Blues", "Greens", "Purples")[stream_index % 3])(
                    display_mask
                )
                alpha = np.clip(
                    (display_mask - args.plot_mask_threshold)
                    / max(1e-6, 1.0 - args.plot_mask_threshold),
                    0.0,
                    1.0,
                )
                rgba[..., 3] = alpha * 0.75
                axis.imshow(rgba, origin="lower", aspect="auto")
                picks, _ = extract_picks(
                    mask,
                    prediction["boxes"][index],
                    args.mask_threshold,
                )
                if picks:
                    channels, times = zip(*picks)
                    axis.scatter(
                        times,
                        channels,
                        s=2,
                        color=colors[stream_index % len(colors)],
                    )
        axis.set_ylabel("Channel index")
    axes[-1].set_xlabel("Time sample index")
    figure.suptitle(os.path.basename(file_name))
    figure.tight_layout()
    keep_labels_inside_axes(figure, box_labels)
    os.makedirs(args.figure_dir, exist_ok=True)
    figure.savefig(
        os.path.join(args.figure_dir, f"{os.path.splitext(os.path.basename(file_name))[0]}.png"),
        dpi=150,
    )
    plt.close(figure)


def build_model(args):
    """Construct a model matching the checkpoint configuration."""
    resolve_backbone(args)
    anchor_sizes, anchor_metadata = resolve_rpn_anchor_sizes(
        args.rpn_anchor_sizes,
        auto=args.rpn_anchor_auto,
        scales_blob=args.rpn_anchor_scales,
        input_height=args.rpn_anchor_samples,
        input_width=args.rpn_anchor_channels,
        resize_scale=args.resize_scale,
        transform_min_size=args.transform_min_size,
        transform_max_size=args.transform_max_size,
    )
    if anchor_metadata:
        print(f"Resolved RPN anchor sizes: {anchor_metadata['anchor_sizes']}")
    return build_dasnet_model(
        model_name="maskrcnn_selectable_fpn",
        backbone_name=args.backbone,
        load_backbone_weights=False,
        num_classes=args.num_classes,
        in_channels=args.image_channels,
        pretrained=False,
        pretrained_backbone=args.backbone_weights == "imagenet",
        trainable_backbone_layers=3,
        target_layer="P4",
        box_roi_pool_size=7,
        mask_roi_pool_size=(42, 42),
        max_size=args.transform_max_size,
        min_size=args.transform_min_size,
        mask_prepare_mode=args.mask_prepare_mode,
        arrival_band_alpha=args.arrival_band_alpha,
        arrival_band_before_frac=args.arrival_band_before_frac,
        arrival_band_after_frac=args.arrival_band_after_frac,
        mask_streams_by_category=args.mask_streams_by_category,
        use_learned_temporal_crop=args.use_learned_temporal_crop,
        loss_temporal_crop_weight=args.loss_temporal_crop_weight,
        temporal_crop_loss_divisor=args.temporal_crop_loss_divisor,
        rpn_anchor_sizes=anchor_sizes,
        rpn_anchor_aspect_ratios=args.rpn_anchor_aspect_ratios,
    )


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    restore_prediction_config(args, checkpoint)
    paths, names = read_input_paths(args.data_list, args.from_raw)
    channel_range = (
        (args.channel_start, args.channel_end) if args.from_raw else None
    )
    dataset = DASInferDataset(
        paths,
        resize_scale=args.resize_scale,
        data_key="data",
        data_is_strain_rate=True,
        display_names=names,
        image_channels=args.image_channels,
        storage_backend=args.storage_backend,
        gcs_key_path=args.gcs_key_path,
        from_raw=args.from_raw,
        channel_range=channel_range,
        denoised_skip_preproc=args.denoised_skip_preproc,
        denoised_apply_normalize=not args.denoised_no_normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    model = build_model(args).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot_figure:
        args.figure_dir = args.figure_dir or os.path.join(args.output_dir, "figures")
    autocast = nullcontext() if args.device == "cpu" else torch.amp.autocast("cuda")
    with torch.inference_mode():
        for images, file_names in loader:
            with autocast:
                outputs = model([image.to(args.device) for image in images])
            for image, file_name, output in zip(images, file_names, outputs):
                prediction = {
                    key: value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else [item.detach().cpu().numpy() for item in value]
                    if isinstance(value, (list, tuple))
                    and all(isinstance(item, torch.Tensor) for item in value)
                    else value
                    for key, value in output.items()
                }
                record = prediction_record(file_name, prediction, args)
                if args.plot_figure:
                    save_prediction_figure(file_name, image.cpu(), prediction, args)
                output_path = os.path.join(
                    args.output_dir, f"{os.path.splitext(os.path.basename(file_name))[0]}.json"
                )
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(record, file, indent=2)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint.")
    parser.add_argument("--data-list", required=True, help="CSV containing input file paths.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default=None, help="Read from checkpoint; explicit mismatches are rejected.")
    parser.add_argument("--backbone-weights", choices=("imagenet", "none"), default=None)
    parser.add_argument("--num-classes", type=int, default=None, help="Read from checkpoint when available.")
    parser.add_argument("--image-channels", choices=(1, 3), type=int, default=1)
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Read raw Acquisition/Raw[0]/RawData HDF5 input and compute strain rate.",
    )
    parser.add_argument("--channel-start", type=int, help="First raw channel, inclusive.")
    parser.add_argument("--channel-end", type=int, help="Last raw channel, exclusive.")
    parser.add_argument(
        "--storage-backend",
        choices=("auto", "local", "gcs"),
        default="auto",
        help="Storage backend for standard or raw HDF5 files.",
    )
    parser.add_argument("--gcs-key-path", help="Optional service-account key for GCS input.")
    parser.add_argument(
        "--denoised-skip-preproc",
        action="store_true",
        help="Use the HDF5 data field directly instead of strain-rate filtering.",
    )
    parser.add_argument(
        "--denoised-no-normalize",
        action="store_true",
        help="Do not normalize data when --denoised-skip-preproc is set.",
    )
    parser.add_argument("--resize-scale", type=float, default=0.5)
    parser.add_argument("--transform-min-size", type=int, default=1422)
    parser.add_argument("--transform-max-size", type=int, default=6000)
    parser.add_argument("--mask-streams-by-class", default="")
    parser.add_argument(
        "--mask-prepare-mode",
        choices=("fixed_crop", "attention_gate", "attention_plus_crop", "arrival_band"),
        default="fixed_crop",
        help="Mask-branch mode used during training.",
    )
    parser.add_argument("--use-learned-temporal-crop", action="store_true")
    parser.add_argument("--loss-temporal-crop-weight", type=float, default=1.0)
    parser.add_argument("--temporal-crop-loss-divisor", type=float, default=1000.0)
    parser.add_argument("--arrival-band-alpha", type=float, default=1.0)
    parser.add_argument("--arrival-band-before-frac", type=float, default=0.5)
    parser.add_argument("--arrival-band-after-frac", type=float, default=0.5)
    parser.add_argument("--rpn-anchor-sizes", default="")
    parser.add_argument("--rpn-anchor-aspect-ratios", default="")
    parser.add_argument("--rpn-anchor-auto", action="store_true")
    parser.add_argument("--rpn-anchor-scales", default="0.125,0.25,0.5,1,1")
    parser.add_argument("--rpn-anchor-samples", type=int, default=4096)
    parser.add_argument("--rpn-anchor-channels", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--min-probability", type=float, default=0.3)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--plot-figure", action="store_true", help="Save prediction overlays as PNG files.")
    parser.add_argument("--figure-dir", help="Directory for prediction figures.")
    parser.add_argument("--plot-mask-threshold", type=float, default=0.25)
    parser.add_argument(
        "--plot-arrival-classes", nargs="+", type=int, default=list(DEFAULT_PLOT_ARRIVAL_CLASSES),
        help="Category IDs whose masks and picks are drawn (default: 2 3 8, Monterey Blue B / Fin / earthquake P-S). All detection boxes remain visible; JSON predictions are unchanged.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


if __name__ == "__main__":
    arguments = get_parser().parse_args()
    arguments.mask_streams_by_category = parse_mask_streams(arguments.mask_streams_by_class)
    arguments.rpn_anchor_sizes = parse_anchor_sizes(arguments.rpn_anchor_sizes)
    arguments.rpn_anchor_aspect_ratios = parse_anchor_ratios(arguments.rpn_anchor_aspect_ratios)
    if arguments.rpn_anchor_auto and arguments.rpn_anchor_sizes is not None:
        raise SystemExit("Specify either --rpn-anchor-sizes or --rpn-anchor-auto.")
    if arguments.from_raw:
        if arguments.channel_start is None or arguments.channel_end is None:
            raise SystemExit("--from-raw requires --channel-start and --channel-end.")
        if arguments.channel_end <= arguments.channel_start:
            raise SystemExit("--channel-end must be greater than --channel-start.")
    main(arguments)
