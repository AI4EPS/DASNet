from .maskrcnn import MaskRCNN
import torch
import torchvision
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torch import nn
import copy
from .backbone_utils import _resnet_customfpn_extractor
from typing import Optional, Tuple


def _adapt_resnet_input_channels(backbone: nn.Module, in_channels: int) -> None:
    """Replace ResNet conv1 for non-RGB inputs.

    New weights are initialized from the (pretrained) RGB kernels averaged over
    the input dim, scaled so the expected response magnitude is preserved.
    """
    old = backbone.conv1
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        mean_w = old.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        new.weight.copy_(mean_w.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    backbone.conv1 = new


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
    **kwargs
) -> MaskRCNN:

    # MaskRCNN-ResNet50-FPN pretrained weights
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    # ResNet pretrained weights
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    # backbone
    if model_name == "maskrcnn_resnet50_selectable_fpn":
        backbone = resnet50(weights=weights_backbone, progress=True, norm_layer=norm_layer)
        if in_channels != 3:
            _adapt_resnet_input_channels(backbone, in_channels)
        backbone = _resnet_customfpn_extractor(backbone, trainable_backbone_layers, target_layer)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if in_channels != 3:
        if pretrained:
            raise ValueError("pretrained MaskRCNN weights are RGB-only; use pretrained_backbone with in_channels != 3")
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

    if model_name == "maskrcnn_resnet50_selectable_fpn":
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

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True))

    return model

