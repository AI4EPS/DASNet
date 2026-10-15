"""Coordinate regressions for learned temporal cropping (CPU, no checkpoints)."""
import unittest
from collections import OrderedDict

import torch
from torchvision.ops import MultiScaleRoIAlign, roi_align

from dasnet.model.learned_temporal_crop import adjust_mask_proposals_temporal
from dasnet.model.das_transform import DASGeneralizedRCNNTransform
from dasnet.model.roiheads import project_masks_on_boxes_with_dif_side


def crop(boxes, logits=None, shape=(120, 80), alpha=1.0, before=0.5, after=0.5):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if logits is None:
        logits = torch.zeros(len(boxes))
    windows, anchors, shapes = adjust_mask_proposals_temporal(
        [boxes], logits, [shape], alpha, before, after
    )
    return windows[0], anchors, shapes


class TemporalCropTests(unittest.TestCase):
    def test_proposal_width_and_image_width_independence(self):
        boxes = [[0, 40, 20, 60], [0, 40, 60, 60]]
        windows, anchors, shapes = crop(boxes)
        torch.testing.assert_close(windows[:, 3] - windows[:, 1], torch.tensor([20., 60.]))
        torch.testing.assert_close((windows[:, 1] + windows[:, 3]) / 2, anchors)
        wide, _, _ = crop(boxes, shape=(120, 800))
        torch.testing.assert_close(windows, wide)
        self.assertEqual(shapes, [(120, 80)])

    def test_start_boundary_does_not_move_other_proposals(self):
        boxes = [[0, 0, 40, 20], [0, 60, 40, 80]]
        windows, anchors, shapes = crop(boxes)
        torch.testing.assert_close(windows[:, [1, 3]], torch.tensor([[-10., 30.], [50., 90.]]))
        alone, _, _ = crop(boxes[1:])
        torch.testing.assert_close(windows[1:], alone)
        self.assertEqual(shapes, [(120, 80)])

    def test_end_boundary_and_asymmetry(self):
        windows, anchors, shapes = crop([[0, 100, 40, 120]], before=0.25, after=0.75)
        torch.testing.assert_close(windows, torch.tensor([[0., 100., 40., 140.]]))
        torch.testing.assert_close(anchors, torch.tensor([110.]))
        self.assertEqual(shapes, [(120, 80)])

    def test_empty_and_mixed_image_batches(self):
        empty = torch.empty((0, 4))
        boxes = torch.tensor([[0., 0., 40., 20.]])
        windows, anchors, shapes = adjust_mask_proposals_temporal(
            [empty, boxes], torch.zeros(1), [(20, 30), (120, 80)], 1., .5, .5
        )
        self.assertEqual(windows[0].shape, (0, 4))
        self.assertEqual(shapes, [(20, 30), (120, 80)])
        self.assertEqual(anchors.tolist(), [10.])
        self.assertEqual(adjust_mask_proposals_temporal([], torch.empty(0), [], 1., .5, .5)[0], [])

    def test_anchor_loss_gradient(self):
        logits = torch.zeros(1, requires_grad=True)
        _, anchors, _ = crop([[0, 0, 40, 20]], logits)
        (anchors - 12).square().mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertNotEqual(logits.grad.item(), 0.)

    def test_outside_feature_samples_are_zero_and_inside_not_shifted(self):
        windows, _, shapes = crop([[0, 0, 40, 20]])
        feature = torch.ones((1, 1, 120, 80), requires_grad=True)
        pool = MultiScaleRoIAlign(['0'], (40, 40), sampling_ratio=2)
        output = pool(OrderedDict([('0', feature)]), [windows], shapes)
        torch.testing.assert_close(output[0, 0, :8], torch.zeros((8, 40)))
        torch.testing.assert_close(output[0, 0, 12:], torch.ones((28, 40)))
        output.sum().backward()
        self.assertEqual(feature.grad[0, 0, 45, 10].item(), 0.)
        self.assertGreater(feature.grad[0, 0, 10, 10].item(), 0.)

    def test_feature_target_and_paste_coordinates_for_two_streams(self):
        # A narrow ridge at each anchor, including both image boundaries.
        boxes = torch.tensor([[10., 0., 50., 20.], [10., 100., 50., 120.]])
        transform = DASGeneralizedRCNNTransform(120, 120, [0.], [1.])
        transform.eval()
        masks, windows_by_stream = [], []
        for logits in (torch.zeros(2), torch.full((2,), 0.8)):
            windows, anchors, shapes = adjust_mask_proposals_temporal(
                [boxes], logits, [(120, 80)], 1., .5, .5
            )
            rows = torch.arange(120).float()[None, :, None]
            gt = torch.exp(-0.5 * ((rows - anchors[:, None, None]) / 1.5).square()).expand(2, 120, 80).contiguous()
            features = gt[:, None].clone().requires_grad_(True)
            rois = torch.cat((torch.arange(2)[:, None], windows[0]), dim=1)
            pooled = roi_align(features, rois, (40, 40), 1., sampling_ratio=-1)
            targets = project_masks_on_boxes_with_dif_side(gt, windows[0], torch.arange(2), 40, 40)
            torch.testing.assert_close(pooled[:, 0], targets)
            pooled.sum().backward()
            self.assertTrue(torch.isfinite(features.grad).all())
            masks.append(pooled.detach())
            windows_by_stream.append(windows[0])
        result = transform.postprocess([{
            'boxes': boxes, 'masks': masks[0], 'mask_streams': masks,
            'mask_paste_boxes_streams': windows_by_stream,
        }], [(120, 80)], [(120, 80)])[0]
        for s, logits in enumerate((torch.zeros(2), torch.full((2,), 0.8))):
            expected = boxes[:, 1] + logits.sigmoid() * (boxes[:, 3] - boxes[:, 1])
            peaks = result['mask_streams'][s][:, 0, :, 30].argmax(dim=1)
            self.assertTrue(((peaks - expected).abs() <= 2).all(), (peaks, expected))

    def test_full_model_training_and_inference(self):
        from dasnet.model.dasnet import build_model
        torch.set_num_threads(1)
        torch.manual_seed(10)
        model = build_model(
            num_classes=2, pretrained=False, pretrained_backbone=False,
            min_size=128, max_size=256, use_learned_temporal_crop=True,
            mask_streams_by_category={1: 2},
            rpn_pre_nms_top_n_train=20, rpn_post_nms_top_n_train=10,
            rpn_pre_nms_top_n_test=20, rpn_post_nms_top_n_test=10,
            box_detections_per_img=4, box_score_thresh=0.0,
        )
        image = torch.randn(3, 256, 128)
        primary = torch.zeros(2, 256, 128, dtype=torch.uint8)
        primary[0, 8:11, 10:110] = 255
        primary[1, 240:243, 10:110] = 255
        secondary = torch.roll(primary, 4, dims=1)
        target = {
            'boxes': torch.tensor([[10., 0., 110., 25.], [10., 230., 110., 255.]]),
            'labels': torch.ones(2, dtype=torch.long), 'masks': primary,
            'mask_streams': [primary, secondary],
            'confident_area_masks': torch.ones_like(primary),
        }
        model.train()
        losses = model([image], [target])
        for key in ('loss_mask', 'loss_mask_stream_1', 'loss_temporal_crop'):
            self.assertIn(key, losses)
            self.assertGreater(losses[key].item(), 0.)
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for head in model.roi_heads.temporal_window_predictors:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            result = model([image])[0]
        self.assertEqual(len(result['mask_streams']), 2)
        self.assertGreater(len(result['boxes']), 0)
        for mask in result['mask_streams']:
            self.assertEqual(mask.shape[-2:], (256, 128))
            self.assertTrue(torch.isfinite(mask).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
