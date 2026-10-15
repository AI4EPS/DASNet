from typing import Dict, List, Optional, Tuple
import torchvision
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import boxes as box_ops, roi_align
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, keypointrcnn_inference, \
    keypointrcnn_loss, maskrcnn_inference
from torchvision.models.detection import _utils as det_utils

import torch
import torch.nn.functional as F
from collections import OrderedDict
import copy
import math


def focal_loss(class_logits, labels, alpha=0.25, gamma=2.0):
    probs = F.softmax(class_logits, dim=-1)
    target_probs = probs[torch.arange(len(labels)), labels]

    # Focal Loss formula
    focal_weight = (1 - target_probs) ** gamma
    loss = -alpha * focal_weight * torch.log(target_probs)

    return loss.mean()


def focal_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the focal loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = focal_loss(class_logits, labels, alpha=0.25, gamma=2.0)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def maskrcnn_inference(x, labels):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Args:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob


def mask_stream_count_for_label(label_id: int, streams_by_category: Dict[int, int]) -> int:
    if not streams_by_category:
        return 1
    return max(1, int(streams_by_category.get(int(label_id), 1)))


def subset_proposals_for_mask_stream(
    prop_per_image: List[Tensor],
    labels_per_image: List[Tensor],
    stream_index: int,
    streams_by_category: Dict[int, int],
) -> Tuple[List[Tensor], List[List[int]]]:
    """Per-image proposal subsets whose class uses more than ``stream_index`` mask streams."""
    prop_sub: List[Tensor] = []
    orig_indices: List[List[int]] = []
    for prop, lab in zip(prop_per_image, labels_per_image):
        if prop.numel() == 0:
            prop_sub.append(prop)
            orig_indices.append([])
            continue
        lab = lab.long()
        keep = [
            j
            for j in range(int(lab.shape[0]))
            if mask_stream_count_for_label(int(lab[j].item()), streams_by_category) > stream_index
        ]
        orig_indices.append(keep)
        if keep:
            idx_t = torch.tensor(keep, device=prop.device, dtype=torch.long)
            prop_sub.append(prop.index_select(0, idx_t))
        else:
            prop_sub.append(prop.new_zeros((0, 4)))
    return prop_sub, orig_indices


def scatter_prop_pool_per_image(
    prop_pool_sub: List[Tensor],
    prop_pool_template_per_image: List[Tensor],
    orig_indices: List[List[int]],
) -> List[Tensor]:
    """Fill full per-image paste boxes; inactive RoIs keep stream-0 (template) boxes."""
    out: List[Tensor] = []
    for pool_tpl, pool_sub, keep in zip(prop_pool_template_per_image, prop_pool_sub, orig_indices):
        if pool_tpl.numel() == 0:
            out.append(pool_tpl)
            continue
        merged = pool_tpl.clone()
        for local_j, global_j in enumerate(keep):
            merged[global_j] = pool_sub[local_j]
        out.append(merged)
    return out


def scatter_mask_logits_to_full(
    logits_sub: Tensor,
    labels_per_image: List[Tensor],
    orig_indices: List[List[int]],
    reference: Tensor,
    inactive_fill: float = -10.0,
) -> Tensor:
    n_total = sum(int(lab.shape[0]) for lab in labels_per_image)
    full = reference.new_full((n_total, *logits_sub.shape[1:]), inactive_fill)
    off = 0
    base = 0
    for lab, keep in zip(labels_per_image, orig_indices):
        for global_j in keep:
            full[base + global_j] = logits_sub[off]
            off += 1
        base += int(lab.shape[0])
    return full


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]


def project_masks_on_boxes_with_dif_side(gt_masks, boxes, matched_idxs, M, N):
    # type: (Tensor, Tensor, Tensor, int, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, N), 1.0)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0
    
    mask_targets = mask_targets / 255.0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss


def resize_confident_area_masks(confident_area_masks, target_size):
    """
    Resize each tensor in the list of confident-area masks to the specified target size.
    """
    resized = []
    for mask in confident_area_masks:
        if mask.dtype != torch.float32:
            mask = mask.float()
        resized_mask = F.interpolate(mask.unsqueeze(1), size=target_size, mode='bilinear', align_corners=False)
        resized_mask = (resized_mask > 0.5).float()
        resized.append(resized_mask.squeeze(1))
    return resized


def confident_area_masks_per_image_from_target(t: dict):
    """Stacked (N,H,W) weights for mask loss / padding; legacy ``attention_masks`` key still accepted."""
    if "confident_area_masks" in t:
        return t["confident_area_masks"]
    return t["attention_masks"]


def normalize_rows(tensor):
    row_max = tensor.max(dim=2, keepdim=True)[0]

    row_max[row_max == 0] = 1

    normalized_tensor = tensor / row_max
    return normalized_tensor


def weighted_confident_area_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, confident_area_masks, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    # discretization_size = mask_logits.shape[-1]
    discretization_size_1 = mask_logits.shape[-2]
    discretization_size_2 = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

    mask_targets = [
        project_masks_on_boxes_with_dif_side(m, p, i, discretization_size_1, discretization_size_2) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_targets = normalize_rows(mask_targets)

    # Gaussian weighted
    kernel_size = 4
    min_threshold = 0.4
    expanded_gaussian_weights = F.avg_pool1d(mask_targets.transpose(1, 2), kernel_size, stride=1, padding=kernel_size // 2)
    expanded_gaussian_weights = expanded_gaussian_weights.transpose(1, 2)
    expanded_gaussian_weights = torch.maximum(expanded_gaussian_weights, torch.tensor(min_threshold, device=expanded_gaussian_weights.device))
    
    # crop gauss weight
    target_height, target_width = mask_targets[0].shape[-2], mask_targets[0].shape[-1]
    expanded_gaussian_weights = expanded_gaussian_weights[:, :target_height, :target_width]

    # Calculate mask loss with no reduction
    mask_logits = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits, mask_targets, reduction='none'
    )

    # Resize confident_area_masks to the same size as gt_masks
    gt_mask_size = (gt_masks[0].shape[-2], gt_masks[0].shape[-1])  
    if confident_area_masks[0].shape[-2:] != gt_mask_size:
        confident_area_masks = resize_confident_area_masks(confident_area_masks, gt_mask_size)
    confident_area_masks = [
        project_masks_on_boxes_with_dif_side(m, p, i, discretization_size_1, discretization_size_2) for m, p, i in zip(confident_area_masks, proposals, mask_matched_idxs)
    ]

    # Apply confident-area weights to the calculated loss
    confident_area_masks = torch.cat(confident_area_masks, dim=0)

    # Check if confident_area_masks is empty
    if confident_area_masks.numel() == 0 or confident_area_masks.sum() == 0:
        return mask_logits.sum() * 0  # Return zero loss in a consistent manner

    weighted_loss = mask_loss * confident_area_masks * expanded_gaussian_weights

    # Sum the weighted loss and normalize by the number of positive samples
    mask_loss = weighted_loss.sum() / (confident_area_masks * expanded_gaussian_weights).sum()
    return mask_loss


def weighted_mask_stream_loss(
    mask_logits,
    proposals,
    gt_masks,
    gt_labels,
    confident_area_masks,
    mask_matched_idxs,
    streams_by_category: Dict[int, int],
    stream_index: int,
):
    """Same weighting as ``weighted_confident_area_maskrcnn_loss`` but only RoIs whose class uses
    more than ``stream_index`` mask streams (see ``streams_by_category``; missing class → 1 stream).
    """
    discretization_size_1 = mask_logits.shape[-2]
    discretization_size_2 = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

    mask_targets = [
        project_masks_on_boxes_with_dif_side(m, p, i, discretization_size_1, discretization_size_2)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels_flat = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_targets = normalize_rows(mask_targets)

    kernel_size = 4
    min_threshold = 0.4
    expanded_gaussian_weights = F.avg_pool1d(mask_targets.transpose(1, 2), kernel_size, stride=1, padding=kernel_size // 2)
    expanded_gaussian_weights = expanded_gaussian_weights.transpose(1, 2)
    expanded_gaussian_weights = torch.maximum(
        expanded_gaussian_weights, torch.tensor(min_threshold, device=expanded_gaussian_weights.device)
    )
    target_height, target_width = mask_targets[0].shape[-2], mask_targets[0].shape[-1]
    expanded_gaussian_weights = expanded_gaussian_weights[:, :target_height, :target_width]

    mask_logits_sel = mask_logits[torch.arange(labels_flat.shape[0], device=labels_flat.device), labels_flat]
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits_sel, mask_targets, reduction="none")

    gt_mask_size = (gt_masks[0].shape[-2], gt_masks[0].shape[-1])
    if confident_area_masks[0].shape[-2:] != gt_mask_size:
        confident_area_masks = resize_confident_area_masks(confident_area_masks, gt_mask_size)
    confident_area_masks = [
        project_masks_on_boxes_with_dif_side(m, p, i, discretization_size_1, discretization_size_2)
        for m, p, i in zip(confident_area_masks, proposals, mask_matched_idxs)
    ]
    confident_area_masks = torch.cat(confident_area_masks, dim=0)

    if confident_area_masks.numel() == 0 or confident_area_masks.sum() == 0:
        return mask_logits.sum() * 0

    nstreams = torch.ones_like(labels_flat, dtype=torch.long)
    for c, nv in streams_by_category.items():
        nstreams[labels_flat == int(c)] = int(nv)
    active = (nstreams > int(stream_index)).to(dtype=mask_loss.dtype).unsqueeze(-1).unsqueeze(-1)
    if active.sum() < 1e-8:
        return mask_logits.sum() * 0

    weighted_loss = mask_loss * confident_area_masks * expanded_gaussian_weights * active
    denom = (confident_area_masks * expanded_gaussian_weights * active).sum()
    if denom < 1e-8:
        return mask_logits.sum() * 0
    return weighted_loss.sum() / denom


def dual_mask_auxiliary_loss(
    mask_logits_aux,
    proposals,
    gt_masks_aux,
    gt_labels,
    mask_matched_idxs,
    dual_class_ids,
    discretization_size_1,
    discretization_size_2,
):
    """BCE on auxiliary mask head; only averaged over RoIs whose matched GT label is in dual_class_ids."""
    if mask_logits_aux.numel() == 0:
        return mask_logits_aux.sum() * 0

    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    labels = torch.cat(labels, dim=0)

    dual_t = torch.zeros_like(labels, dtype=torch.bool)
    for c in dual_class_ids:
        dual_t |= labels == c
    if not dual_t.any():
        return mask_logits_aux.sum() * 0

    mask_targets = [
        project_masks_on_boxes_with_dif_side(m, p, i, discretization_size_1, discretization_size_2)
        for m, p, i in zip(gt_masks_aux, proposals, mask_matched_idxs)
    ]
    mask_targets = torch.cat(mask_targets, dim=0).float()
    mask_targets = normalize_rows(mask_targets)

    logits = mask_logits_aux[:, 0] if mask_logits_aux.shape[1] >= 1 else mask_logits_aux.squeeze(1)
    loss = F.binary_cross_entropy_with_logits(logits, mask_targets, reduction="none")
    loss = loss.flatten(1).mean(dim=1)
    return loss[dual_t].mean()


def confident_area_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, confident_area_masks, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_targets = normalize_rows(mask_targets)

    # Calculate mask loss with no reduction
    mask_logits = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits, mask_targets, reduction='none'
    )

    # Resize confident_area_masks to the same size as gt_masks
    gt_mask_size = (gt_masks[0].shape[-2], gt_masks[0].shape[-1])
    if confident_area_masks[0].shape[-2:] != gt_mask_size:
        confident_area_masks = resize_confident_area_masks(confident_area_masks, gt_mask_size)

    confident_area_masks = resize_confident_area_masks(confident_area_masks, gt_mask_size)
    confident_area_masks = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(confident_area_masks, proposals, mask_matched_idxs)
    ]

    # Apply confident-area weights to the calculated loss
    confident_area_masks = torch.cat(confident_area_masks, dim=0)

    # Check if confident_area_masks is empty
    if confident_area_masks.numel() == 0:
        return mask_logits.sum() * 0  # Return zero loss in a consistent manner

    weighted_loss = mask_loss * confident_area_masks
    
    # Sum the weighted loss and normalize by the number of positive samples
    mask_loss = weighted_loss.sum() / confident_area_masks.sum()
    return mask_loss


def focal_confident_area_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, confident_area_masks, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_targets = normalize_rows(mask_targets)

    # Calculate mask loss with no reduction
    mask_logits = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]

    alpha=0.25
    gamma=2.0
    mask_probs = torch.sigmoid(mask_logits)
    bce_loss = F.binary_cross_entropy_with_logits(
        mask_logits, mask_targets, reduction='none'
    )
    pt = mask_probs * mask_targets + (1 - mask_probs) * (1 - mask_targets)
    focal_weight = alpha * (1 - pt) ** gamma

    mask_loss = focal_weight * bce_loss

    # Resize confident_area_masks to the same size as gt_masks
    gt_mask_size = (gt_masks[0].shape[-2], gt_masks[0].shape[-1])
    if confident_area_masks[0].shape[-2:] != gt_mask_size:
        confident_area_masks = resize_confident_area_masks(confident_area_masks, gt_mask_size)

    confident_area_masks = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(confident_area_masks, proposals, mask_matched_idxs)
    ]

    # Apply confident-area weights to the calculated loss
    confident_area_masks = torch.cat(confident_area_masks, dim=0)

    # Check if confident_area_masks is empty
    if confident_area_masks.numel() == 0 or confident_area_masks.sum() == 0:
        return mask_logits.sum() * 0  # Return zero loss in a consistent manner

    weighted_loss = mask_loss * confident_area_masks

    # Sum the weighted loss and normalize by the number of positive samples
    mask_loss = weighted_loss.sum() / confident_area_masks.sum()
    return mask_loss


# Deprecated aliases (older notebooks / forks)
resize_attention_masks = resize_confident_area_masks
weighted_attention_maskrcnn_loss = weighted_confident_area_maskrcnn_loss
attention_maskrcnn_loss = confident_area_maskrcnn_loss
focal_attention_maskrcnn_loss = focal_confident_area_maskrcnn_loss


def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid


def _onnx_heatmaps_to_keypoints(
    maps, maps_i, roi_map_width, roi_map_height, widths_i, heights_i, offset_x_i, offset_y_i
):
    num_keypoints = torch.scalar_tensor(maps.size(1), dtype=torch.int64)

    width_correction = widths_i / roi_map_width
    height_correction = heights_i / roi_map_height

    roi_map = F.interpolate(
        maps_i[:, None], size=(int(roi_map_height), int(roi_map_width)), mode="bicubic", align_corners=False
    )[:, 0]

    w = torch.scalar_tensor(roi_map.size(2), dtype=torch.int64)
    pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

    x_int = pos % w
    y_int = (pos - x_int) // w

    x = (torch.tensor(0.5, dtype=torch.float32) + x_int.to(dtype=torch.float32)) * width_correction.to(
        dtype=torch.float32
    )
    y = (torch.tensor(0.5, dtype=torch.float32) + y_int.to(dtype=torch.float32)) * height_correction.to(
        dtype=torch.float32
    )

    xy_preds_i_0 = x + offset_x_i.to(dtype=torch.float32)
    xy_preds_i_1 = y + offset_y_i.to(dtype=torch.float32)
    xy_preds_i_2 = torch.ones(xy_preds_i_1.shape, dtype=torch.float32)
    xy_preds_i = torch.stack(
        [
            xy_preds_i_0.to(dtype=torch.float32),
            xy_preds_i_1.to(dtype=torch.float32),
            xy_preds_i_2.to(dtype=torch.float32),
        ],
        0,
    )

    # TODO: simplify when indexing without rank will be supported by ONNX
    base = num_keypoints * num_keypoints + num_keypoints + 1
    ind = torch.arange(num_keypoints)
    ind = ind.to(dtype=torch.int64) * base
    end_scores_i = (
        roi_map.index_select(1, y_int.to(dtype=torch.int64))
        .index_select(2, x_int.to(dtype=torch.int64))
        .view(-1)
        .index_select(0, ind.to(dtype=torch.int64))
    )

    return xy_preds_i, end_scores_i


@torch.jit._script_if_tracing
def _onnx_heatmaps_to_keypoints_loop(
    maps, rois, widths_ceil, heights_ceil, widths, heights, offset_x, offset_y, num_keypoints
):
    xy_preds = torch.zeros((0, 3, int(num_keypoints)), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((0, int(num_keypoints)), dtype=torch.float32, device=maps.device)

    for i in range(int(rois.size(0))):
        xy_preds_i, end_scores_i = _onnx_heatmaps_to_keypoints(
            maps, maps[i], widths_ceil[i], heights_ceil[i], widths[i], heights[i], offset_x[i], offset_y[i]
        )
        xy_preds = torch.cat((xy_preds.to(dtype=torch.float32), xy_preds_i.unsqueeze(0).to(dtype=torch.float32)), 0)
        end_scores = torch.cat(
            (end_scores.to(dtype=torch.float32), end_scores_i.to(dtype=torch.float32).unsqueeze(0)), 0
        )
    return xy_preds, end_scores


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_keypoints = maps.shape[1]

    if torchvision._is_tracing():
        xy_preds, end_scores = _onnx_heatmaps_to_keypoints_loop(
            maps,
            rois,
            widths_ceil,
            heights_ceil,
            widths,
            heights,
            offset_x,
            offset_y,
            torch.scalar_tensor(num_keypoints, dtype=torch.int64),
        )
        return xy_preds.permute(0, 2, 1), end_scores

    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = F.interpolate(
            maps[i][:, None], size=(roi_map_height, roi_map_width), mode="bicubic", align_corners=False
        )[:, 0]
        # roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

        x_int = pos % w
        y_int = torch.div(pos - x_int, w, rounding_mode="floor")
        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[torch.arange(num_keypoints, device=roi_map.device), y_int, x_int]

    return xy_preds.permute(0, 2, 1), end_scores


def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    N, K, H, W = keypoint_logits.shape
    if H != W:
        raise ValueError(
            f"keypoint_logits height and width (last two elements of shape) should be equal. Instead got H = {H} and W = {W}"
        )
    discretization_size = H
    heatmaps = []
    valid = []
    for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kp, proposals_per_image, discretization_size)
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.where(valid)[0]

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it sepaartely
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    keypoint_logits = keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
    return keypoint_loss


def keypointrcnn_inference(x, boxes):
    # type: (Tensor, List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
    kp_probs = []
    kp_scores = []

    boxes_per_image = [box.size(0) for box in boxes]
    x2 = x.split(boxes_per_image, dim=0)

    for xx, bb in zip(x2, boxes):
        kp_prob, scores = heatmaps_to_keypoints(xx, bb)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)

    return kp_probs, kp_scores


def _onnx_expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half = w_half.to(dtype=torch.float32) * scale
    h_half = h_half.to(dtype=torch.float32) * scale

    boxes_exp0 = x_c - w_half
    boxes_exp1 = y_c - h_half
    boxes_exp2 = x_c + w_half
    boxes_exp3 = y_c + h_half
    boxes_exp = torch.stack((boxes_exp0, boxes_exp1, boxes_exp2, boxes_exp3), 1)
    return boxes_exp


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily for paste_mask_in_image
def expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    if torchvision._is_tracing():
        return _onnx_expand_boxes(boxes, scale)
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


@torch.jit.unused
def expand_masks_tracing_scale(M, padding):
    # type: (int, int) -> float
    return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)


def expand_masks(mask, padding):
    # type: (Tensor, int) -> Tuple[Tensor, float]
    M = mask.shape[-1]
    if torch._C._get_tracing_state():  # could not import is_tracing(), not sure why
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = F.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w):
    # type: (Tensor, Tensor, int, int) -> Tensor
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]
    return im_mask


def _onnx_paste_mask_in_image(mask, box, im_h, im_w):
    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)

    w = box[2] - box[0] + one
    h = box[3] - box[1] + one
    w = torch.max(torch.cat((w, one)))
    h = torch.max(torch.cat((h, one)))

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))

    # Resize mask
    mask = F.interpolate(mask, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
    y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))

    unpaded_im_mask = mask[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]

    # TODO : replace below with a dynamic padding when support is added in ONNX

    # pad y
    zeros_y0 = torch.zeros(y_0, unpaded_im_mask.size(1))
    zeros_y1 = torch.zeros(im_h - y_1, unpaded_im_mask.size(1))
    concat_0 = torch.cat((zeros_y0, unpaded_im_mask.to(dtype=torch.float32), zeros_y1), 0)[0:im_h, :]
    # pad x
    zeros_x0 = torch.zeros(concat_0.size(0), x_0)
    zeros_x1 = torch.zeros(concat_0.size(0), im_w - x_1)
    im_mask = torch.cat((zeros_x0, concat_0, zeros_x1), 1)[:, :im_w]
    return im_mask


@torch.jit._script_if_tracing
def _onnx_paste_masks_in_image_loop(masks, boxes, im_h, im_w):
    res_append = torch.zeros(0, im_h, im_w)
    for i in range(masks.size(0)):
        mask_res = _onnx_paste_mask_in_image(masks[i][0], boxes[i], im_h, im_w)
        mask_res = mask_res.unsqueeze(0)
        res_append = torch.cat((res_append, mask_res))
    return res_append


def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    if torchvision._is_tracing():
        return _onnx_paste_masks_in_image_loop(
            masks, boxes, torch.scalar_tensor(im_h, dtype=torch.int64), torch.scalar_tensor(im_w, dtype=torch.int64)
        )[:, None]
    res = [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret


def expand_gt_and_confident_area_masks(gt_masks, confident_area_masks, image_shapes):
    """
    Expand gt_masks and confident_area_masks to match the expanded image size for each image in the batch.
    :param gt_masks: List[Tensor], each Tensor of shape (H, W), representing ground truth masks.
    :param confident_area_masks: List[Tensor], each Tensor of shape (H, W), per-image stacked instance weights.
    :param image_shapes: List[Tuple[int, int]], expanded image dimensions for each image in the batch.
    :return: Expanded gt_masks and confident_area_masks.
    """
    expanded_gt_masks = []
    expanded_ca_masks = []

    # Iterate over each image in the batch
    for gt_mask, ca_mask, (H_exp, W_exp) in zip(gt_masks, confident_area_masks, image_shapes):
        # Get original sizes
        H_gt, W_gt = gt_mask.shape[-2:]

        # Compute required padding
        pad_height = max(0, H_exp - H_gt)
        pad_width = max(0, W_exp - W_gt)

        # Apply padding to gt_masks
        expanded_gt_mask = torch.nn.functional.pad(
            gt_mask,
            (0, pad_width, 0, pad_height),  # (left, right, top, bottom)
            mode="constant",
            value=0  # Fill with 0
        )
        expanded_gt_masks.append(expanded_gt_mask)

        # Apply padding to confident_area_masks
        expanded_ca_mask = torch.nn.functional.pad(
            ca_mask,
            (0, pad_width, 0, pad_height),  # (left, right, top, bottom)
            mode="constant",
            value=0  # Fill with 0
        )
        expanded_ca_masks.append(expanded_ca_mask)

    return expanded_gt_masks, expanded_ca_masks


expand_gt_and_attention_masks = expand_gt_and_confident_area_masks


def expand_stacked_instance_masks(gt_masks_list, image_shapes):
    """
    Pad each image's stacked instance masks (N, H, W) to match expanded (H_exp, W_exp).
    Same padding rule as expand_gt_and_confident_area_masks.
    """
    out = []
    for gt_mask, (H_exp, W_exp) in zip(gt_masks_list, image_shapes):
        H_gt, W_gt = gt_mask.shape[-2:]
        pad_height = max(0, H_exp - H_gt)
        pad_width = max(0, W_exp - W_gt)
        out.append(
            torch.nn.functional.pad(gt_mask, (0, pad_width, 0, pad_height), mode="constant", value=0)
        )
    return out


def gt_masks_for_stream_target(target: dict, stream_index: int) -> torch.Tensor:
    """Stacked GT masks for arrival geometry / loss stream ``stream_index`` (0 = primary ``masks``)."""
    streams = target.get("mask_streams")
    if streams is not None and stream_index < len(streams):
        return streams[stream_index]
    if stream_index == 0:
        return target["masks"]
    return torch.zeros_like(target["masks"])


def calculate_expansion_scale(original_shapes, expanded_shapes, expand_vertical=True):
    """
    Calculate the expansion scale for each image (for the dimension that was expanded).
    :param original_shapes: List[Tuple[int, int]], original (H_img, W_img). DAS: H=time, W=channel.
    :param expanded_shapes: List[Tuple[int, int]], expanded (H_exp, W_exp).
    :param expand_vertical: If True, we expanded H (time) -> return scale_y = H_exp/H_orig.
    :return: List[float], expansion scales per image.
    """
    scales = []
    for (H_orig, W_orig), (H_exp, W_exp) in zip(original_shapes, expanded_shapes):
        if expand_vertical:
            scales.append(H_exp / H_orig)
        else:
            scales.append(W_exp / W_orig)
    return scales


def expand_image_shapes_fixed(image_shapes, alpha=1.0, use_space_to_expand_time=True):
    """
    Expand image_shapes for mask ROI. DAS convention: (H_img, W_img) = (time, channel).
    :param image_shapes: List[Tuple[int, int]], (H_img, W_img).
    :param alpha: scale factor.
    :param use_space_to_expand_time: If True, expand time by channel scale: additional_height = W_img * alpha.
    :return: Expanded image_shapes.
    """
    expanded_image_shapes = []
    for H_img, W_img in image_shapes:
        if use_space_to_expand_time:
            # time dim expanded by channel (space) width
            additional_height = int(W_img * alpha)
            expanded_height = H_img + additional_height
            expanded_image_shapes.append((expanded_height, W_img))
        else:
            additional_width = int(H_img * alpha)
            expanded_width = W_img + additional_width
            expanded_image_shapes.append((H_img, expanded_width))
    return expanded_image_shapes


def expand_features(features, scales, expand_vertical=True):
    """
    Expand features for each image. DAS: feature (B, C, H_feat, W_feat) with H=time, W=channel.
    :param expand_vertical: If True, expand H (time) dimension; else expand W (channel).
    """
    expanded_features = OrderedDict()

    for level_name, feature_map in features.items():
        B, C, H_feat, W_feat = feature_map.shape
        expanded_feature_maps = []

        for b in range(B):
            scale = scales[b]
            if expand_vertical:
                expanded_size = int(H_feat * scale)
                if expanded_size <= H_feat:
                    expanded_feature_maps.append(feature_map[b:b + 1])
                    continue
                padding_height = expanded_size - H_feat
                expanded_map = torch.nn.functional.pad(
                    feature_map[b:b + 1],
                    (0, 0, 0, padding_height),  # (left, right, top, bottom)
                    mode="constant",
                    value=0
                )
            else:
                expanded_size = int(W_feat * scale)
                if expanded_size <= W_feat:
                    expanded_feature_maps.append(feature_map[b:b + 1])
                    continue
                padding_width = expanded_size - W_feat
                expanded_map = torch.nn.functional.pad(
                    feature_map[b:b + 1],
                    (0, padding_width, 0, 0),
                    mode="constant",
                    value=0
                )
            expanded_feature_maps.append(expanded_map)

        # Batched (B, C, H, W) must share H (or W) across dim=0; per-image scales can differ
        # (e.g. crop / learned temporal) → align with bottom/right padding before cat.
        if expand_vertical:
            max_h = max(x.shape[2] for x in expanded_feature_maps)
            aligned = []
            for x in expanded_feature_maps:
                pad_h = max_h - x.shape[2]
                if pad_h > 0:
                    x = torch.nn.functional.pad(x, (0, 0, 0, pad_h), mode="constant", value=0)
                aligned.append(x)
            expanded_feature_maps = aligned
        else:
            max_w = max(x.shape[3] for x in expanded_feature_maps)
            aligned = []
            for x in expanded_feature_maps:
                pad_w = max_w - x.shape[3]
                if pad_w > 0:
                    x = torch.nn.functional.pad(x, (pad_w, 0, 0, 0), mode="constant", value=0)
                aligned.append(x)
            expanded_feature_maps = aligned

        expanded_features[level_name] = torch.cat(expanded_feature_maps, dim=0)

    return expanded_features


def crop_mask_proposals_and_expand(mask_proposals, image_shapes, alpha=1.0, enable_crop=True, use_space_to_crop_time=True):
    """
    Crop mask proposals and expand image_shapes. Box: [x_min, y_min, x_max, y_max] with x=channel, y=time (DAS).
    :param alpha: scale factor.
    :param use_space_to_crop_time: If True, time crop by channel (space) extent: new_height = width * alpha.
    """
    if (not enable_crop) or (alpha is None):
        return mask_proposals, image_shapes

    expanded_image_shapes = expand_image_shapes_fixed(image_shapes, alpha, use_space_to_expand_time=use_space_to_crop_time)

    cropped_proposals = []
    for proposal in mask_proposals:
        x_min, y_min, x_max, y_max = proposal.unbind(dim=1)
        if use_space_to_crop_time:
            # time crop length from channel (space) width
            width = x_max - x_min
            new_height = alpha * width
            y_max_cropped = y_min + new_height
            max_height = max(H_exp for H_exp, _ in expanded_image_shapes)
            y_max_cropped = torch.clamp(y_max_cropped, max=max_height)
            cropped_proposal = torch.stack([x_min, y_min, x_max, y_max_cropped], dim=1)
        else:
            height = y_max - y_min
            new_width = alpha * height
            x_max_cropped = x_min + new_width
            max_width = max(W_exp for _, W_exp in expanded_image_shapes)
            x_max_cropped = torch.clamp(x_max_cropped, max=max_width)
            cropped_proposal = torch.stack([x_min, y_min, x_max_cropped, y_max], dim=1)
        cropped_proposals.append(cropped_proposal)

    return cropped_proposals, expanded_image_shapes


class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        # crop feature (time expansion for DAS mask ROI)
        v_ref: float = 1000,
        spacing: float = 5.1,
        time_step: float = 1/200,
        enable_crop: bool = True,
        # mask ROI preparation: see doc/ATTENTION_DUAL_MASK.md
        mask_prepare_mode: str = "arrival_band",
        attention_gate_floor: float = 0.05,
        attention_gate_constrain_masks: bool = False,
        attention_gate_mask_dilate_px: int = 0,
        dual_mask_class_ids: Tuple[int, ...] = (),
        enable_dual_mask_aux: bool = False,
        mask_streams_by_category: Optional[Dict[int, int]] = None,
        skip_inactive_mask_streams_at_inference: bool = True,
        # arrival_band gate (from GT masks); see doc/ATTENTION_DUAL_MASK.md
        arrival_band_alpha: float = 1.0,
        arrival_band_before_frac: float = 0.25,
        arrival_band_after_frac: float = 0.75,
        arrival_gate_class_ids: Tuple[int, ...] = (),
        enable_learned_mask_gate: bool = False,
        loss_mask_gate_weight: float = 1.0,
        arrival_centroid_use_confident_area: bool = True,
        record_inference_mask_gate: bool = False,
        use_learned_temporal_crop: bool = False,
        loss_temporal_crop_weight: float = 1.0,
        temporal_crop_loss_divisor: float = 100.0,
        fpn_out_channels: int = 256,
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

        self.v_ref = v_ref
        self.spacing = spacing
        self.time_step = time_step
        self.enable_crop = enable_crop

        self.mask_prepare_mode = str(mask_prepare_mode)
        self.attention_gate_floor = float(attention_gate_floor)
        self.attention_gate_constrain_masks = bool(attention_gate_constrain_masks)
        self.attention_gate_mask_dilate_px = int(attention_gate_mask_dilate_px)
        if self.attention_gate_mask_dilate_px < 0:
            raise ValueError("attention_gate_mask_dilate_px must be >= 0")
        self.dual_mask_class_ids = tuple(int(x) for x in dual_mask_class_ids)
        self.enable_dual_mask_aux = bool(enable_dual_mask_aux) and len(self.dual_mask_class_ids) > 0

        _msc: Dict[int, int] = {}
        if mask_streams_by_category:
            _msc = {int(k): int(v) for k, v in mask_streams_by_category.items()}
        elif self.enable_dual_mask_aux:
            for c in self.dual_mask_class_ids:
                _msc[int(c)] = 2
        self.mask_streams_by_category = _msc
        self.max_mask_streams = max(1, max(self.mask_streams_by_category.values()) if self.mask_streams_by_category else 1)
        self.skip_inactive_mask_streams_at_inference = bool(skip_inactive_mask_streams_at_inference)

        mp = mask_predictor
        self.mask_predictors = None if mp is None else nn.ModuleList([mp])
        if self.mask_predictors is not None:
            while len(self.mask_predictors) < self.max_mask_streams:
                self.mask_predictors.append(copy.deepcopy(self.mask_predictors[0]))
        self.mask_predictor = self.mask_predictors[0] if self.mask_predictors is not None else None

        self.arrival_band_alpha = float(arrival_band_alpha)
        self.arrival_band_before_frac = float(arrival_band_before_frac)
        self.arrival_band_after_frac = float(arrival_band_after_frac)
        if self.arrival_band_before_frac <= 0 or self.arrival_band_after_frac <= 0:
            raise ValueError("arrival_band_before_frac and arrival_band_after_frac must be > 0")
        self.arrival_gate_class_ids = tuple(int(x) for x in arrival_gate_class_ids)

        self.enable_learned_mask_gate = bool(enable_learned_mask_gate)
        self.loss_mask_gate_weight = float(loss_mask_gate_weight)
        self.arrival_centroid_use_confident_area = bool(arrival_centroid_use_confident_area)
        self.record_inference_mask_gate = bool(record_inference_mask_gate)
        self.use_learned_temporal_crop = bool(use_learned_temporal_crop)
        self.loss_temporal_crop_weight = float(loss_temporal_crop_weight)
        _tcd = float(temporal_crop_loss_divisor)
        if _tcd <= 0:
            raise ValueError("temporal_crop_loss_divisor must be > 0")
        self.temporal_crop_loss_divisor = _tcd
        self.fpn_out_channels = int(fpn_out_channels)
        if self.use_learned_temporal_crop and self.enable_learned_mask_gate:
            raise ValueError(
                "use_learned_temporal_crop and enable_learned_mask_gate cannot both be True "
                "(learned temporal crop replaces pre-pool geometry; disable mask_gate_students)."
            )
        if self.enable_learned_mask_gate:
            from .learned_mask_gate import MaskGateStudent

            self.mask_gate_students = nn.ModuleList(
                [MaskGateStudent(in_channels=256, hidden=128) for _ in range(self.max_mask_streams)]
            )
        else:
            self.mask_gate_students = None
        self.mask_gate_student = self.mask_gate_students[0] if self.mask_gate_students is not None else None

        if self.use_learned_temporal_crop:
            from .learned_temporal_crop import TemporalWindowPredictor

            sp = int(self.box_roi_pool.output_size[0])
            self.temporal_window_predictors = nn.ModuleList(
                [
                    TemporalWindowPredictor(
                        in_channels=self.fpn_out_channels,
                        spatial_size=sp,
                        hidden=512,
                    )
                    for _ in range(self.max_mask_streams)
                ]
            )
        else:
            self.temporal_window_predictors = None

        if self.mask_prepare_mode not in ("fixed_crop", "attention_gate", "attention_plus_crop", "arrival_band"):
            raise ValueError(
                f"mask_prepare_mode must be fixed_crop|attention_gate|attention_plus_crop|arrival_band, got {self.mask_prepare_mode!r}"
            )

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def _teacher_temporal_anchor_batch(
        self,
        targets,
        pos_matched_idxs,
        proposals_per_image,
        stream_s: int,
        image_shapes,
    ):
        """Teacher time centroids (same rule as arrival_band) for Smooth L1 on predicted anchors."""
        from .mask_roi_conditioner import _arrival_band_region_for_instance_j

        vals: List[float] = []
        valid: List[bool] = []
        dev = torch.device("cpu")
        for p in proposals_per_image:
            if p.numel() > 0:
                dev = p.device
                break
        for i, t in enumerate(targets):
            if proposals_per_image[i].numel() == 0:
                continue
            mids = pos_matched_idxs[i]
            n = proposals_per_image[i].shape[0]
            H_i, W_i = image_shapes[i]
            mfull = gt_masks_for_stream_target(t, stream_s)
            for k in range(n):
                mj = int(mids[k].item())
                if mj < 0:
                    vals.append(0.0)
                    valid.append(False)
                    continue
                reg = _arrival_band_region_for_instance_j(
                    t,
                    mj,
                    H_i,
                    W_i,
                    self.arrival_band_alpha,
                    self.arrival_band_before_frac,
                    self.arrival_band_after_frac,
                    self.arrival_gate_class_ids,
                    use_confident_area_for_centroid=self.arrival_centroid_use_confident_area,
                    instance_masks=mfull,
                )
                if reg is None:
                    vals.append(0.0)
                    valid.append(False)
                else:
                    vals.append(float(reg["t_mean"]))
                    valid.append(True)
        if not vals:
            e = torch.empty(0, device=dev, dtype=torch.float32)
            return e, e.to(dtype=torch.bool)
        return (
            torch.tensor(vals, device=dev, dtype=torch.float32),
            torch.tensor(valid, device=dev, dtype=torch.bool),
        )

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")
        if self.has_mask():
            if not all(["masks" in t for t in targets]):
                raise ValueError("Every element of targets should have a masks key")

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels


    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                from .mask_roi_conditioner import aggregate_image_attention_gate, apply_feature_attention_gate

                orig_shapes = list(image_shapes)
                mask_logits_list: List[torch.Tensor] = []
                gate_logits_list: List[Optional[torch.Tensor]] = []
                prop_pool_per_stream: List[List[torch.Tensor]] = []
                temporal_crop_terms: List[torch.Tensor] = []
                record_ta = (
                    (not self.training)
                    and self.use_learned_temporal_crop
                    and self.temporal_window_predictors is not None
                )
                ta_by_img_stream: Optional[List[List[Optional[torch.Tensor]]]] = (
                    [[None] * self.max_mask_streams for _ in range(len(result))] if record_ta else None
                )

                labels_per_img_infer = [r["labels"] for r in result] if not self.training else None

                for s in range(self.max_mask_streams):
                    feat_mask = OrderedDict((k, v) for k, v in features.items())
                    img_shapes_m = list(image_shapes)
                    prop_m = mask_proposals
                    orig_idx_stream: Optional[List[List[int]]] = None
                    use_subset_infer = (
                        (not self.training)
                        and s >= 1
                        and bool(self.mask_streams_by_category)
                        and self.skip_inactive_mask_streams_at_inference
                        and labels_per_img_infer is not None
                    )
                    if use_subset_infer:
                        prop_sub, orig_idx_stream = subset_proposals_for_mask_stream(
                            prop_m,
                            labels_per_img_infer,
                            s,
                            self.mask_streams_by_category,
                        )
                        n_sub = sum(int(p.shape[0]) for p in prop_sub)
                        n_all = sum(int(p.shape[0]) for p in prop_m)
                        if n_sub == 0:
                            ref_logits = mask_logits_list[0]
                            mask_logits_list.append(
                                ref_logits.new_full(
                                    (n_all, *ref_logits.shape[1:]),
                                    -10.0,
                                )
                            )
                            gate_logits_list.append(None)
                            prop_pool_per_stream.append(
                                [prop_pool_per_stream[0][img_i].clone() for img_i in range(len(result))]
                            )
                            if record_ta and ta_by_img_stream is not None:
                                for img_i, r in enumerate(result):
                                    nb = int(r["boxes"].shape[0])
                                    ta_by_img_stream[img_i][s] = r["boxes"].new_zeros((nb,))
                            continue
                        if n_sub < n_all:
                            prop_m = prop_sub
                        else:
                            orig_idx_stream = None

                    if self.mask_prepare_mode in ("attention_gate", "attention_plus_crop"):
                        if self.training and targets is not None:
                            gates = aggregate_image_attention_gate(
                                targets,
                                img_shapes_m,
                                self.attention_gate_floor,
                                constrain_with_gt_masks=self.attention_gate_constrain_masks,
                                mask_dilate_px=self.attention_gate_mask_dilate_px,
                            )
                            feat_mask = apply_feature_attention_gate(feat_mask, img_shapes_m, gates)

                    elif self.mask_prepare_mode == "arrival_band":
                        if self.training and targets is not None:
                            from .mask_roi_conditioner import (
                                build_arrival_band_gates_from_gt_masks,
                                apply_feature_attention_gate,
                            )

                            m_override = [gt_masks_for_stream_target(t, s) for t in targets]
                            gates = build_arrival_band_gates_from_gt_masks(
                                targets,
                                img_shapes_m,
                                alpha=self.arrival_band_alpha,
                                before_frac=self.arrival_band_before_frac,
                                after_frac=self.arrival_band_after_frac,
                                class_ids=self.arrival_gate_class_ids,
                                floor=self.attention_gate_floor,
                                use_confident_area_for_centroid=self.arrival_centroid_use_confident_area,
                                gt_masks_override_per_image=m_override,
                            )
                            feat_mask = apply_feature_attention_gate(feat_mask, img_shapes_m, gates)

                    prop_pool = prop_m
                    _dev = next(iter(feat_mask.values())).device
                    y_anchor_cat = torch.zeros(0, device=_dev, dtype=torch.float32)
                    if self.use_learned_temporal_crop and self.temporal_window_predictors is not None:
                        from .learned_temporal_crop import adjust_mask_proposals_temporal

                        npos = sum(int(p.shape[0]) for p in prop_m)
                        if npos > 0:
                            tw = self.box_roi_pool(feat_mask, prop_m, img_shapes_m)
                            logits_tw = self.temporal_window_predictors[s](tw)
                            prop_pool, y_anchor_cat, exp_shapes = adjust_mask_proposals_temporal(
                                prop_m,
                                logits_tw,
                                img_shapes_m,
                                self.arrival_band_alpha,
                                self.arrival_band_before_frac,
                                self.arrival_band_after_frac,
                            )
                            # Temporal windows stay in original image coordinates.
                            # Do not expand/shift features or alter the RoIAlign scale.
                            img_shapes_m = list(exp_shapes)
                            if (
                                self.training
                                and targets is not None
                                and pos_matched_idxs is not None
                                and y_anchor_cat is not None
                                and y_anchor_cat.numel() > 0
                            ):
                                y_t, vmask = self._teacher_temporal_anchor_batch(
                                    targets, pos_matched_idxs, prop_m, s, orig_shapes
                                )
                                if vmask.any():
                                    temporal_crop_terms.append(
                                        F.smooth_l1_loss(
                                            y_anchor_cat[vmask],
                                            y_t[vmask].to(dtype=y_anchor_cat.dtype),
                                            reduction="mean",
                                        )
                                    )
                            elif record_ta and npos > 0 and y_anchor_cat.numel() > 0:
                                sizes = [int(p.shape[0]) for p in prop_m]
                                if orig_idx_stream is None:
                                    for img_i, ch in enumerate(torch.split(y_anchor_cat, sizes, dim=0)):
                                        ta_by_img_stream[img_i][s] = ch.detach()
                                else:
                                    for img_i, (ch_img, keep) in enumerate(
                                        zip(torch.split(y_anchor_cat, sizes, dim=0), orig_idx_stream)
                                    ):
                                        nb = int(labels_per_img_infer[img_i].shape[0])
                                        full_ta = mask_proposals[img_i].new_zeros(
                                            (nb,), dtype=y_anchor_cat.dtype
                                        )
                                        for local_j, global_j in enumerate(keep):
                                            full_ta[global_j] = ch_img[local_j]
                                        ta_by_img_stream[img_i][s] = full_ta.detach()

                    do_crop = (
                        self.mask_prepare_mode in ("fixed_crop", "attention_plus_crop", "arrival_band")
                        and self.enable_crop
                        and not self.use_learned_temporal_crop
                    )
                    if do_crop:
                        crop_alpha = (self.v_ref * self.time_step) / self.spacing
                        prop_pool, img_shapes_m = crop_mask_proposals_and_expand(
                            prop_pool,
                            img_shapes_m,
                            alpha=crop_alpha,
                            enable_crop=True,
                            use_space_to_crop_time=True,
                        )
                        scales = calculate_expansion_scale(orig_shapes, img_shapes_m, expand_vertical=True)
                        if any(float(s) > 1.0 + 1e-6 for s in scales):
                            s_max = max(float(s) for s in scales)
                            scales_u = [s_max] * len(orig_shapes)
                            feat_mask = expand_features(feat_mask, scales_u, expand_vertical=True)
                            img_shapes_m = [
                                (int(math.ceil(float(ho) * s_max)), int(wo)) for ho, wo in orig_shapes
                            ]
                        else:
                            feat_mask = expand_features(feat_mask, scales, expand_vertical=True)

                    if orig_idx_stream is not None:
                        prop_pool = scatter_prop_pool_per_image(
                            prop_pool,
                            prop_pool_per_stream[0],
                            orig_idx_stream,
                        )

                    prop_pool_per_stream.append(prop_pool)

                    mask_features = self.mask_roi_pool(feat_mask, prop_pool, img_shapes_m)
                    mask_features = self.mask_head(mask_features)
                    gate_logits_s: Optional[torch.Tensor] = None
                    if self.mask_gate_students is not None:
                        gate_logits_s = self.mask_gate_students[s](mask_features)
                        gate = torch.sigmoid(gate_logits_s)
                        mask_features = mask_features * gate
                    mask_logits_s = self.mask_predictors[s](mask_features)
                    if orig_idx_stream is not None and labels_per_img_infer is not None:
                        mask_logits_s = scatter_mask_logits_to_full(
                            mask_logits_s,
                            labels_per_img_infer,
                            orig_idx_stream,
                            mask_logits_list[0],
                        )
                        if gate_logits_s is not None:
                            gate_logits_s = scatter_mask_logits_to_full(
                                gate_logits_s,
                                labels_per_img_infer,
                                orig_idx_stream,
                                gate_logits_list[0],
                                inactive_fill=-10.0,
                            )
                    mask_logits_list.append(mask_logits_s)
                    gate_logits_list.append(gate_logits_s)

                if ta_by_img_stream is not None:
                    for img_i, r in enumerate(result):
                        streams: List[torch.Tensor] = []
                        for s in range(self.max_mask_streams):
                            t = ta_by_img_stream[img_i][s]
                            if t is None:
                                nb = int(r["boxes"].shape[0])
                                t = r["boxes"].new_zeros((nb,), dtype=torch.float32)
                            streams.append(t)
                        r["temporal_anchor_streams"] = streams
                        r["temporal_anchors"] = streams[0]

                if (
                    not self.training
                    and self.record_inference_mask_gate
                    and self.mask_gate_students is not None
                    and all(gl is not None for gl in gate_logits_list)
                ):
                    labs = [r["labels"] for r in result]
                    bpi = [x.shape[0] for x in labs]
                    streams_per_image: List[List[torch.Tensor]] = []
                    for gl in gate_logits_list:
                        gp = torch.sigmoid(gl).detach()
                        streams_per_image.append(list(gp.split(bpi, dim=0)))
                    for i, r in enumerate(result):
                        r["mask_gate"] = streams_per_image[0][i]
                        if len(streams_per_image) > 1:
                            r["mask_gate_streams"] = [streams_per_image[s][i] for s in range(len(streams_per_image))]

                mask_logits = mask_logits_list[0]
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or not mask_logits_list:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_labels = [t["labels"] for t in targets]
                confident_area_masks = [confident_area_masks_per_image_from_target(t) for t in targets]

                loss_gate_acc = 0.0
                n_gate_terms = 0
                for s in range(self.max_mask_streams):
                    gt_s = [gt_masks_for_stream_target(t, s) for t in targets]
                    gt_s, confident_area_masks_s = expand_gt_and_confident_area_masks(
                        gt_s, confident_area_masks, img_shapes_m
                    )
                    prop_use = prop_pool_per_stream[s]
                    lm = weighted_mask_stream_loss(
                        mask_logits_list[s],
                        prop_use,
                        gt_s,
                        gt_labels,
                        confident_area_masks_s,
                        pos_matched_idxs,
                        self.mask_streams_by_category,
                        s,
                    )
                    if s == 0:
                        loss_mask["loss_mask"] = lm
                    else:
                        loss_mask[f"loss_mask_stream_{s}"] = lm

                    if self.mask_gate_students is not None and gate_logits_list[s] is not None:
                        from .learned_mask_gate import (
                            build_teacher_gates_for_positive_proposals,
                            mask_gate_distillation_loss,
                        )

                        m_teacher = [gt_masks_for_stream_target(t, s) for t in targets]
                        gl = gate_logits_list[s]
                        # Teacher must match student gate map resolution (mask_head), not mask logits (predictor upsample).
                        th, v = build_teacher_gates_for_positive_proposals(
                            targets,
                            img_shapes_m,
                            prop_use,
                            pos_matched_idxs,
                            (int(gl.shape[-2]), int(gl.shape[-1])),
                            alpha=self.arrival_band_alpha,
                            before_frac=self.arrival_band_before_frac,
                            after_frac=self.arrival_band_after_frac,
                            class_ids=self.arrival_gate_class_ids,
                            use_confident_area_for_centroid=self.arrival_centroid_use_confident_area,
                            teacher_masks_per_image=m_teacher,
                        )
                        th = th.detach().to(device=gl.device, dtype=gl.dtype)
                        v = v.to(device=gl.device, dtype=gl.dtype)
                        if th.shape[0] == gl.shape[0] and th.numel() > 0:
                            loss_gate_acc += mask_gate_distillation_loss(gl, th, v)
                            n_gate_terms += 1
                if n_gate_terms > 0:
                    loss_mask["loss_mask_gate"] = self.loss_mask_gate_weight * (loss_gate_acc / n_gate_terms)

                if temporal_crop_terms:
                    loss_mask["loss_temporal_crop"] = (
                        self.loss_temporal_crop_weight
                        * torch.stack(temporal_crop_terms).mean()
                        / self.temporal_crop_loss_divisor
                    )

            else:
                labels = [r["labels"] for r in result]
                masks_probs_per_stream = [maskrcnn_inference(ml, labels) for ml in mask_logits_list]
                for i, r in enumerate(result):
                    r["masks"] = masks_probs_per_stream[0][i]
                    if self.max_mask_streams > 1:
                        r["mask_streams"] = [masks_probs_per_stream[s][i] for s in range(self.max_mask_streams)]
                    # Learned temporal crop: mask logits were pooled on ``prop_pool`` per stream, not on
                    # detection boxes. ``DASGeneralizedRCNNTransform`` must paste each stream with its own
                    # RoI box (same coords as pre-postprocess ``boxes``); otherwise all streams map into
                    # the full detection window and secondary structures (e.g. aux ridge) collapse visually.
                    if self.use_learned_temporal_crop:
                        r["mask_paste_boxes_streams"] = [
                            prop_pool_per_stream[s][i] for s in range(self.max_mask_streams)
                        ]

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses
