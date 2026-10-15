import unittest
from unittest.mock import patch
import torch
import train


class TrainingEntrypointTests(unittest.TestCase):
    def test_nested_streams_move_to_device(self):
        target={'boxes':torch.ones(1,4),'mask_streams':(torch.ones(1,4,4),torch.ones(1,4,4)),
                'extra':[torch.ones(1)],'image_id':12}
        moved=train.move_targets([target],'meta')[0]
        self.assertEqual(moved['boxes'].device.type,'meta')
        self.assertTrue(all(m.device.type=='meta' for m in moved['mask_streams']))
        self.assertIsInstance(moved['mask_streams'],tuple)
        self.assertEqual(moved['extra'][0].device.type,'meta')
        self.assertEqual(moved['image_id'],12)

    def test_monterey_augmentation_and_geometry(self):
        args=train.get_parser().parse_args(['--data-path','data','--anno-path','train.json',
            '--val-data-path','data','--val-anno-path','val.json','--output-dir','out',
            '--num-classes','10','--noise-csv','noise.csv'])
        args.mask_streams_by_category={8:2}
        with patch.object(train,'DASTrainDataset') as dataset:
            train.build_datasets(args)
        first,second=dataset.call_args_list
        self.assertFalse(first.kwargs['enable_stack_event'])
        self.assertFalse(second.kwargs['enable_stack_event'])
        self.assertTrue(first.kwargs['synthetic_noise'])
        self.assertFalse(second.kwargs['synthetic_noise'])
        self.assertTrue(first.kwargs['enable_vflip'])
        self.assertFalse(second.kwargs['enable_vflip'])
        self.assertEqual(args.resize_scale,0.5)
        self.assertEqual((args.transform_min_size,args.transform_max_size),(1422,6000))
