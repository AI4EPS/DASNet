"""Detection transform: extend torchvision postprocess to paste optional mask-branch gates."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes, resize_keypoints


class DASGeneralizedRCNNTransform(GeneralizedRCNNTransform):
    """Same as ``GeneralizedRCNNTransform``, plus pasting ``mask_gate`` / ``mask_gate_streams`` like ``masks``."""

    def postprocess(
        self,
        result: List[dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[dict[str, Tensor]]:
        if self.training:
            return result
        # Inlined from GeneralizedRCNNTransform.postprocess so we can paste masks with per-stream
        # RoI boxes when ``mask_paste_boxes_streams`` is set (learned temporal crop).
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes_rs = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes_rs

            paste_streams = pred.pop("mask_paste_boxes_streams", None)

            if "masks" in pred:
                masks = pred["masks"]
                if paste_streams is not None and len(paste_streams) > 0:
                    paste_b0 = resize_boxes(paste_streams[0], im_s, o_im_s)
                else:
                    paste_b0 = boxes_rs
                result[i]["masks"] = paste_masks_in_image(masks, paste_b0, o_im_s)

            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                result[i]["keypoints"] = resize_keypoints(keypoints, im_s, o_im_s)

            mss = pred.get("mask_streams")
            if mss is not None:
                if paste_streams is not None and len(paste_streams) == len(mss):
                    result[i]["mask_streams"] = [
                        paste_masks_in_image(m, resize_boxes(paste_streams[s], im_s, o_im_s), o_im_s)
                        for s, m in enumerate(mss)
                    ]
                else:
                    result[i]["mask_streams"] = [paste_masks_in_image(m, boxes_rs, o_im_s) for m in mss]

            if "mask_gate" in pred:
                result[i]["mask_gate"] = paste_masks_in_image(pred["mask_gate"], boxes_rs, o_im_s)
            mgs = pred.get("mask_gate_streams")
            if mgs is not None:
                result[i]["mask_gate_streams"] = [paste_masks_in_image(g, boxes_rs, o_im_s) for g in mgs]
        return result
