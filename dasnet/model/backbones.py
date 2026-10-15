"""Swin-Tiny and ResNet50 adapters for the existing 128-channel DASNet FPN."""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .backbone_utils import _resnet_customfpn_extractor, CustomFeaturePyramidNetwork


SUPPORTED_BACKBONES = ("swin_t", "resnet50")


class SwinStageBody(nn.Module):
    """Convert NHWC stages to NCHW and preserve the legacy FPN strides.

    Swin stages have strides 4/8/16/32. Pooling their outputs by two gives
    8/16/32/64, matching DASNet's ResNet adapter without changing Swin blocks.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        outputs = OrderedDict()
        for index, block in enumerate(self.features):
            x = block(x)
            if index in (1, 3, 5, 7):
                stage = x.permute(0, 3, 1, 2).contiguous()
                outputs[str(len(outputs))] = F.avg_pool2d(stage, 2, 2, ceil_mode=True)
        return outputs


class SwinBackboneWithFPN(nn.Module):
    def __init__(self, features, target_layer):
        super().__init__()
        self.body = SwinStageBody(features)
        self.fpn = CustomFeaturePyramidNetwork(
            [96, 192, 384, 768], 128, target_layer=target_layer,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = 128

    def forward(self, x):
        return self.fpn(self.body(x))


def _adapt_input_conv(old, in_channels):
    if in_channels < 1:
        raise ValueError("in_channels must be positive")
    if in_channels == 3:
        return old
    new = nn.Conv2d(
        in_channels, old.out_channels, kernel_size=old.kernel_size,
        stride=old.stride, padding=old.padding, bias=old.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True)
                         .repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


def build_backbone(name="resnet50", *, pretrained=True, load_weights=True,
                   in_channels=3, trainable_layers=3, target_layer="P4"):
    """Keep checkpoint normalization but skip initialization downloads on restore.

    Swin trains all stages. ResNet retains the existing trainable_layers policy.
    The lower-level default remains ResNet50 for existing build_model callers;
    new command-line training defaults to Swin-Tiny.
    """
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {SUPPORTED_BACKBONES}")
    if name == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained and load_weights else None
        model = models.swin_t(weights=weights, progress=True)
        model.features[0][0] = _adapt_input_conv(model.features[0][0], in_channels)
        return SwinBackboneWithFPN(model.features, target_layer)
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained and load_weights else None
    trainable_layers = _validate_trainable_layers(pretrained, trainable_layers, 5, 3)
    norm = misc_nn_ops.FrozenBatchNorm2d if pretrained else nn.BatchNorm2d
    body = models.resnet50(weights=weights, progress=True, norm_layer=norm)
    body.conv1 = _adapt_input_conv(body.conv1, in_channels)
    return _resnet_customfpn_extractor(body, trainable_layers, target_layer)
