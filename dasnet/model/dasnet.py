from .maskrcnn import MaskRCNN
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torch import nn
import copy
from .backbones import build_backbone
from typing import Optional, Tuple


def build_model(
    model_name: str = "maskrcnn_resnet50_selectable_fpn",
    num_classes: int = None,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    target_layer: str = 'P4',
    box_roi_pool_size: int = 7,
    mask_roi_pool_size: Tuple[int, int] = (42, 42),
    in_channels: int = 3,
    rpn_anchor_sizes: Optional[Tuple[Tuple[int, ...], ...]] = None,
    rpn_anchor_aspect_ratios: Optional[Tuple[Tuple[float, ...], ...]] = None,
    backbone_name: str = "resnet50",
    load_backbone_weights: bool = True,
    **kwargs
) -> MaskRCNN:

    if model_name not in ("maskrcnn_resnet50_selectable_fpn", "maskrcnn_selectable_fpn"):
        raise ValueError(f"Unsupported model_name: {model_name}")
    if pretrained:
        raise ValueError("Full torchvision MaskRCNN weights do not match this custom 128-channel FPN; use pretrained_backbone=True instead.")
    if num_classes is None:
        num_classes = 91
    backbone = build_backbone(
        backbone_name, pretrained=pretrained_backbone, load_weights=load_backbone_weights,
        in_channels=in_channels, trainable_layers=trainable_backbone_layers,
        target_layer=target_layer)

    if in_channels != 3:
        # transform normalization must match channel count; data is already
        # z-scored so use the averaged ImageNet stats per channel
        kwargs.setdefault("image_mean", [0.449] * in_channels)
        kwargs.setdefault("image_std", [0.226] * in_channels)

    model = MaskRCNN(backbone, num_classes=num_classes, **kwargs)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    mp = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    model.roi_heads.mask_predictors = nn.ModuleList([mp])
    while len(model.roi_heads.mask_predictors) < model.roi_heads.max_mask_streams:
        model.roi_heads.mask_predictors.append(copy.deepcopy(model.roi_heads.mask_predictors[0]))
    model.roi_heads.mask_predictor = model.roi_heads.mask_predictors[0]

    model.roi_heads.box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['1', '2', '3', '4'],
        output_size=box_roi_pool_size,
        sampling_ratio=2,
    )
    model.roi_heads.mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['1', '2', '3', '4'],
        output_size=mask_roi_pool_size,
        sampling_ratio=2,
    )

    if rpn_anchor_sizes is None:
        if target_layer == 'P2':
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        elif target_layer == 'P3':
            anchor_sizes = ((64,), (64,), (128,), (256,), (512,))
        elif target_layer == 'P4':
            anchor_sizes = ((128,), (128,), (128,), (256,), (512,))
        elif target_layer == 'P5':
            anchor_sizes = ((256,), (256,), (256,), (256,), (512,))
        else:
            anchor_sizes = ((512,), (512,), (512,), (512,), (512,))
    else:
        anchor_sizes = rpn_anchor_sizes

    if rpn_anchor_aspect_ratios is None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    elif len(rpn_anchor_aspect_ratios) == 1:
        aspect_ratios = rpn_anchor_aspect_ratios * len(anchor_sizes)
    elif len(rpn_anchor_aspect_ratios) == len(anchor_sizes):
        aspect_ratios = rpn_anchor_aspect_ratios
    else:
        raise ValueError(
            "rpn_anchor_aspect_ratios must have either one group or the same "
            f"number of groups as rpn_anchor_sizes ({len(anchor_sizes)}), got {len(rpn_anchor_aspect_ratios)}"
        )
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    num_anchors_per_location = anchor_generator.num_anchors_per_location()
    if len(set(num_anchors_per_location)) != 1:
        raise ValueError(
            "All FPN levels must have the same number of anchors per location; "
            f"got {num_anchors_per_location}. Adjust sizes/ratios so len(sizes[i]) * len(ratios[i]) is constant."
        )
    model.rpn.anchor_generator = anchor_generator
    num_anchors = int(num_anchors_per_location[0])
    if model.rpn.head.cls_logits.out_channels != num_anchors:
        model.rpn.head = RPNHead(model.backbone.out_channels, num_anchors)

    return model

