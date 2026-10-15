"""Backbone integration: feature contract, backward, masks and checkpoint restore."""
import argparse
import gc
import unittest
from unittest.mock import patch
import torch
from dasnet.model.backbones import build_backbone, SUPPORTED_BACKBONES
from dasnet.model.dasnet import build_model
from dasnet.model.checkpoint_config import resolve_backbone, restore_prediction_config
import predict


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_feature_contract_all_backbones(self):
        for name in SUPPORTED_BACKBONES:
            with self.subTest(name=name):
                model = build_backbone(name, pretrained=True, load_weights=False).eval()
                with torch.inference_mode():
                    features = model(torch.zeros(1, 3, 256, 256))
                self.assertEqual(list(features), ['0', '1', '2', '3', 'pool'])
                self.assertEqual([tuple(x.shape) for x in features.values()],
                                 [(1,128,s,s) for s in [32,16,8,4,2]])
                del model, features
                gc.collect()

    def test_full_model_backward_and_inference(self):
        for name in ['resnet50', 'swin_t']:
            with self.subTest(name=name):
                torch.manual_seed(10)
                model = build_model(backbone_name=name, num_classes=2,
                    pretrained_backbone=True, load_backbone_weights=False,
                    min_size=128, max_size=256, use_learned_temporal_crop=True,
                    mask_streams_by_category={1:2},
                    rpn_pre_nms_top_n_train=20, rpn_post_nms_top_n_train=10,
                    rpn_pre_nms_top_n_test=20, rpn_post_nms_top_n_test=10,
                    box_detections_per_img=4, box_score_thresh=0.0)
                image = torch.randn(3,256,128)
                mask = torch.zeros(2,256,128,dtype=torch.uint8)
                mask[0,8:11,10:110]=255;mask[1,240:243,10:110]=255
                target={'boxes':torch.tensor([[10.,0.,110.,25.],[10.,230.,110.,255.]]),
                        'labels':torch.ones(2,dtype=torch.long),'masks':mask,
                        'mask_streams':[mask,torch.roll(mask,4,dims=1)],
                        'confident_area_masks':torch.ones_like(mask)}
                model.train();losses=model([image],[target]);loss=sum(losses.values())
                self.assertTrue(torch.isfinite(loss));loss.backward()
                for key in ['loss_classifier','loss_box_reg','loss_mask','loss_mask_stream_1','loss_temporal_crop']:
                    self.assertIn(key,losses)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.body.parameters()))
                for head in model.roi_heads.temporal_window_predictors:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters()))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                model.eval()
                with torch.inference_mode():out=model([image])[0]
                self.assertGreater(len(out['boxes']),0)
                self.assertEqual(len(out['mask_streams']),2)
                for stream in out['mask_streams']:
                    self.assertEqual(stream.shape[-2:],(256,128));self.assertTrue(torch.isfinite(stream).all())
                del model, losses, loss, out, image, mask, target
                gc.collect()

    def test_checkpoint_defaults_restore_and_reject_mismatch(self):
        fresh = resolve_backbone(argparse.Namespace(backbone=None, backbone_weights=None))
        self.assertEqual((fresh.backbone, fresh.backbone_weights), ('swin_t', 'imagenet'))
        args=argparse.Namespace(backbone=None,backbone_weights=None)
        resolve_backbone(args,{'args':argparse.Namespace()})
        self.assertEqual((args.backbone,args.backbone_weights),('resnet50','imagenet'))
        for name in ['swin_t']:
            cp={'args':{'backbone':name,'backbone_weights':'none'}}
            resolved=resolve_backbone(argparse.Namespace(backbone=None,backbone_weights=None),cp)
            self.assertEqual((resolved.backbone,resolved.backbone_weights),(name,'none'))
            with self.assertRaisesRegex(ValueError,'Checkpoint backbone='):
                resolve_backbone(argparse.Namespace(backbone='resnet50'),cp)
        with self.assertRaises(ValueError):build_backbone('not-a-backbone')

    def test_checkpoint_predict_roundtrip_without_download(self):
        for name in ['resnet50','swin_t']:
            with self.subTest(name=name):
                # Full training-state construction with FrozenBN for legacy ResNet.
                torch.manual_seed(7)
                original=build_model(backbone_name=name,num_classes=3,
                    pretrained_backbone=True,load_backbone_weights=False,
                    min_size=128,max_size=256,use_learned_temporal_crop=True,
                    mask_streams_by_category={2:2})
                cp={'model':original.state_dict(),'args':argparse.Namespace(
                    backbone=name,backbone_weights='imagenet',num_classes=3,
                    image_channels=3,resize_scale=.5,transform_min_size=128,
                    transform_max_size=256,mask_streams_by_category={2:2},
                    use_learned_temporal_crop=True,rpn_anchor_sizes=None,
                    rpn_anchor_aspect_ratios=None,rpn_anchor_auto=False)}
                args=predict.get_parser().parse_args(['--checkpoint','unused','--data-list','unused','--output-dir','unused'])
                args.mask_streams_by_category={}
                restore_prediction_config(args,cp)
                self.assertEqual(args.mask_streams_by_category,{2:2})
                with patch('torch.hub.download_url_to_file',side_effect=AssertionError('Unexpected checkpoint initialization download')):
                    restored=predict.build_model(args).eval()
                restored.load_state_dict(cp['model'],strict=True)
                original.eval()
                with torch.inference_mode():
                    image=torch.randn(1,3,128,256)
                    before=original.backbone(image);after=restored.backbone(image)
                    for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
                del original, restored, cp, before, after
                gc.collect()

    def test_one_channel_swin_input(self):
        for name in ['swin_t']:
            model=build_backbone(name,pretrained=True,load_weights=False,in_channels=1).eval()
            with torch.inference_mode():out=model(torch.zeros(1,1,128,256))
            self.assertEqual(out['0'].shape,(1,128,16,32))
            del model,out
