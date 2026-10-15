"""Train DASNet on local HDF5 data and COCO annotations."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time
from contextlib import nullcontext

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

import utils
from dasnet.data.das import DASTrainDataset
from dasnet.data.das_crop import DASCropTrainDataset
from dasnet.model.dasnet import build_model
from dasnet.model.backbones import SUPPORTED_BACKBONES
from dasnet.model.checkpoint_config import resolve_backbone
from dasnet.model.rpn_anchor_config import resolve_rpn_anchor_sizes


def load_bins_config(path: str) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    """Load optional SNR-bin filtering settings from a JSON file."""
    with open(path, encoding="utf-8") as file:
        config = json.load(file)
    allowed = {
        int(category): tuple(bounds)
        for category, bounds in config.get("allowed_bins", {}).items()
    }
    minimum = {
        int(category): int(value)
        for category, value in config.get("min_keep_bins", {}).items()
    }
    return allowed, minimum


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


def learning_rate_lambda(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """Return a warm-up followed by cosine-decay learning rate."""
    if warmup and step < warmup:
        return ((max_lr - min_lr) * step / warmup + min_lr) / max_lr
    if step >= total:
        return min_lr / max_lr
    ratio = (step - warmup) / max(total - warmup, 1)
    value = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))
    return value / max_lr


def move_targets(targets, device):
    """Move nested mask streams together with all other target tensors."""
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value
    return [move(target) for target in targets]


def make_loss_logger():
    logger = utils.MetricLogger(delimiter="  ")
    logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    logger.add_meter("loss", utils.SmoothedValue(window_size=20, fmt="{value:.6f}"))
    return logger


def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, args):
    """Run one optimization epoch."""
    model.train()
    logger = make_loss_logger()
    optimizer.zero_grad(set_to_none=True)
    autocast = (
        nullcontext()
        if args.device == "cpu"
        else torch.amp.autocast(device_type="cuda", dtype=args.amp_dtype)
    )

    for step, (images, targets) in enumerate(
        logger.log_every(loader, args.print_freq, f"Epoch [{epoch}]")
    ):
        images = [image.to(args.device) for image in images]
        targets = move_targets(targets, args.device)
        with autocast:
            losses = sum(model(images, targets).values())

        if not torch.isfinite(losses):
            raise RuntimeError(f"Non-finite training loss: {losses.item()}")
        scaled_loss = losses / args.accumulation_steps
        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()

        if (step + 1) % args.accumulation_steps == 0 or step + 1 == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        logger.update(loss=losses.item(), lr=optimizer.param_groups[0]["lr"])
    logger.synchronize_between_processes()
    return {name: meter.global_avg for name, meter in logger.meters.items()}


@torch.inference_mode()
def validate(model, loader, args):
    """Compute validation loss without modifying model parameters."""
    model.train()
    logger = make_loss_logger()
    for images, targets in logger.log_every(loader, args.print_freq, "Validation"):
        images = [image.to(args.device) for image in images]
        targets = move_targets(targets, args.device)
        losses = sum(model(images, targets).values())
        logger.update(loss=losses.item())
    logger.synchronize_between_processes()
    return logger.loss.global_avg


def build_datasets(args):
    """Build train and validation datasets using the original HDF5 readers."""
    allowed_bins, min_keep_bins = (
        load_bins_config(args.allowed_bins_json)
        if args.allowed_bins_json
        else ({}, {})
    )
    stream_options = {
        "mask_streams_by_category": args.mask_streams_by_category or None,
    }
    crop_options = {
        "crop_channels": args.crop_channels,
        "crop_samples": args.crop_samples,
        "resize_scale": args.resize_scale,
        "image_channels": args.image_channels,
        "min_picks_anchor": args.min_picks_anchor,
        "min_picks_keep": args.min_picks_keep,
    }
    if args.use_crop_dataset:
        train = DASCropTrainDataset(
            ann_path=args.anno_path,
            root_dir=args.data_path,
            allowed_bins=allowed_bins,
            min_keep_bins=min_keep_bins,
            synthetic_noise=bool(args.noise_csv),
            noise_csv=args.noise_csv,
            enable_vflip=True,
            **crop_options,
            **stream_options,
        )
        validation = DASCropTrainDataset(
            ann_path=args.val_anno_path,
            root_dir=args.val_data_path,
            synthetic_noise=False,
            enable_vflip=False,
            **crop_options,
            **stream_options,
        )
    else:
        if args.image_channels != 3:
            raise ValueError("Full-segment training requires --image-channels 3.")
        train = DASTrainDataset(
            ann_path=args.anno_path,
            root_dir=args.data_path,
            allowed_bins=allowed_bins,
            min_keep_bins=min_keep_bins,
            resize_scale=args.resize_scale,
            synthetic_noise=bool(args.noise_csv),
            noise_csv=args.noise_csv,
            enable_stack_event=False,
            enable_vflip=True,
            **stream_options,
        )
        validation = DASTrainDataset(
            ann_path=args.val_anno_path,
            root_dir=args.val_data_path,
            resize_scale=args.resize_scale,
            synthetic_noise=False,
            enable_stack_event=False,
            enable_vflip=False,
            **stream_options,
        )
    return train, validation


def build_dataloaders(train_dataset, validation_dataset, args):
    """Build local or distributed data loaders."""
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    validation_sampler = (
        DistributedSampler(validation_dataset, shuffle=False) if args.distributed else None
    )
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_batch,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=True,
        **options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        drop_last=False,
        **options,
    )
    return train_loader, validation_loader, train_sampler


def build_dasnet(args, *, load_backbone_weights=True):
    """Construct the training model from command-line configuration."""
    resolve_backbone(args)
    anchor_sizes, anchor_metadata = resolve_rpn_anchor_sizes(
        args.rpn_anchor_sizes,
        auto=args.rpn_anchor_auto,
        scales_blob=args.rpn_anchor_scales,
        input_height=args.crop_samples if args.use_crop_dataset else None,
        input_width=args.crop_channels if args.use_crop_dataset else None,
        resize_scale=args.resize_scale,
        transform_min_size=args.transform_min_size,
        transform_max_size=args.transform_max_size,
    )
    if anchor_metadata and utils.is_main_process():
        print(f"Resolved RPN anchor sizes: {anchor_metadata['anchor_sizes']}")
    return build_model(
        model_name="maskrcnn_selectable_fpn",
        backbone_name=args.backbone,
        load_backbone_weights=load_backbone_weights,
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
        mask_prepare_mode="fixed_crop",
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


def save_checkpoint(path, model, optimizer, scheduler, epoch, args):
    """Save all state required to resume training or run inference."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        },
        path,
    )


def main(args):
    utils.init_distributed_mode(args)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    resolve_backbone(args, checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed + utils.get_rank())
    device = torch.device(args.device)
    args.amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda" and args.amp_dtype == torch.float16)

    train_dataset, validation_dataset = build_datasets(args)
    train_loader, validation_loader, train_sampler = build_dataloaders(
        train_dataset, validation_dataset, args
    )
    model = build_dasnet(args, load_backbone_weights=checkpoint is None).to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_lambda(
            step,
            args.learning_rate,
            args.learning_rate * args.min_learning_rate_ratio,
            args.warmup_epochs * len(train_loader),
            total_steps,
        ),
    )
    start_epoch = 0
    if args.resume:
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1

    best_loss = float("inf")
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, args)
        loss = validate(model, validation_loader, args)
        if utils.is_main_process():
            state_path = os.path.join(args.output_dir, "checkpoint.pth")
            save_checkpoint(state_path, model_without_ddp, optimizer, scheduler, epoch, args)
            if loss < best_loss:
                best_loss = loss
                save_checkpoint(
                    os.path.join(args.output_dir, "model_best.pth"),
                    model_without_ddp,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                )
            with open(os.path.join(args.output_dir, "metrics.jsonl"), "a", encoding="utf-8") as log:
                log.write(json.dumps({"epoch": epoch, "train": train_metrics, "validation_loss": loss}) + "\n")
            print(f"Validation loss: {loss:.6f}")
    if utils.is_main_process():
        duration = datetime.timedelta(seconds=int(time.time() - started))
        print(f"Training completed in {duration}.")


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Directory containing training HDF5 files.")
    parser.add_argument("--anno-path", required=True, help="Training COCO annotation JSON.")
    parser.add_argument("--val-data-path", required=True, help="Directory containing validation HDF5 files.")
    parser.add_argument("--val-anno-path", required=True, help="Validation COCO annotation JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints.")
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default=None,
                        help="Default swin_t; inherited from checkpoint on resume.")
    parser.add_argument("--backbone-weights", choices=("imagenet", "none"), default=None,
                        help="Default imagenet; none initializes the backbone randomly.")
    parser.add_argument("--num-classes", type=int, required=True, help="Class count including background.")
    parser.add_argument("--image-channels", choices=(1, 3), type=int, default=3)
    parser.add_argument(
        "--use-crop-dataset",
        action="store_true",
        help="Read random windows from large HDF5 segments.",
    )
    parser.add_argument("--crop-channels", type=int, default=2048)
    parser.add_argument("--crop-samples", type=int, default=4096)
    parser.add_argument("--min-picks-anchor", type=int, default=64)
    parser.add_argument("--min-picks-keep", type=int, default=8)
    parser.add_argument("--resize-scale", type=float, default=0.5)
    parser.add_argument("--allowed-bins-json", help="Optional SNR-bin configuration JSON.")
    parser.add_argument("--noise-csv", help="Optional CSV with a File Path column for noise augmentation.")
    parser.add_argument("--mask-streams-by-class", default="", help="Mask stream counts, for example 1:2.")
    parser.add_argument("--use-learned-temporal-crop", action="store_true")
    parser.add_argument("--loss-temporal-crop-weight", type=float, default=1.0)
    parser.add_argument("--temporal-crop-loss-divisor", type=float, default=1000.0)
    parser.add_argument("--arrival-band-alpha", type=float, default=1.0)
    parser.add_argument("--arrival-band-before-frac", type=float, default=0.5)
    parser.add_argument("--arrival-band-after-frac", type=float, default=0.5)
    parser.add_argument("--transform-min-size", type=int, default=1422)
    parser.add_argument("--transform-max-size", type=int, default=6000)
    parser.add_argument("--rpn-anchor-sizes", default="")
    parser.add_argument("--rpn-anchor-aspect-ratios", default="")
    parser.add_argument("--rpn-anchor-auto", action="store_true")
    parser.add_argument("--rpn-anchor-scales", default="0.125,0.25,0.5,1,1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", help="Checkpoint to resume.")
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=402)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--dist-url", default="env://")
    return parser


if __name__ == "__main__":
    arguments = get_parser().parse_args()
    arguments.mask_streams_by_category = parse_mask_streams(arguments.mask_streams_by_class)
    arguments.rpn_anchor_sizes = parse_anchor_sizes(arguments.rpn_anchor_sizes)
    arguments.rpn_anchor_aspect_ratios = parse_anchor_ratios(arguments.rpn_anchor_aspect_ratios)
    if arguments.rpn_anchor_auto and arguments.rpn_anchor_sizes is not None:
        raise SystemExit("Specify either --rpn-anchor-sizes or --rpn-anchor-auto.")
    if arguments.rpn_anchor_auto and not arguments.use_crop_dataset:
        raise SystemExit("--rpn-anchor-auto requires --use-crop-dataset.")
    main(arguments)
