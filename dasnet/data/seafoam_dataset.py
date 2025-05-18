from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torch
from torchvision.transforms import functional as F
import numpy as np
import os
import h5py, random
import pandas as pd

def interpolate_line_segments_int(points):
    line_points = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        max_delta = max(abs(dx), abs(dy))
        if max_delta == 0:
            return points

        step_x = dx / max_delta
        step_y = dy / max_delta

        line_segment = [(int(start[0] + step_x * j), int(start[1] + step_y * j)) for j in range(max_delta + 1)]

        line_points.extend(line_segment)

    unique_points = list(set(line_points))
    unique_points.sort(key=lambda x: (x[0], x[1]))

    return unique_points


def gaussian_line(x_center, std_dev, amplitude, length):
    x = np.arange(length)
    gaussian = amplitude * np.exp(-(x - x_center)**2 / (2 * std_dev**2))
    return gaussian


def normalize(data, pclip=100):
    clipVal = np.percentile(np.absolute(data), pclip)
    norm_data = np.clip(data, -clipVal, clipVal)
    norm_data = (norm_data + clipVal) / (2 * clipVal) * 255
    return norm_data


def block_normalize(data, block_height, block_width, range_min=0, range_max=255):

    height, width = data.shape
    normalized_data = np.zeros_like(data, dtype=np.float32)
    
    for i in range(0, height, block_height):
        for j in range(0, width, block_width):
            block = data[i:i + block_height, j:j + block_width]
            
            block_min = np.min(block)
            block_max = np.max(block)
            
            if block_max - block_min == 0:
                normalized_block = np.zeros_like(block)
            else:
                normalized_block = (block - block_min) / (block_max - block_min) * (range_max - range_min) + range_min
            
            normalized_data[i:i + block_height, j:j + block_width] = normalized_block
    
    return normalized_data


class DASNet__SeaFOAMDataset(Dataset):
    def __init__(self, ann_path, root_dir, device='cpu', target_size=(948, 4000), synthetic_noise=False):
        self.coco = COCO(ann_path)
        self.root_dir = root_dir
        self.device = device
        self.ids = list(self.coco.imgs.keys())
        self.target_size = target_size
        self.synthetic_noise = synthetic_noise

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        # Load the image
        path = coco.loadImgs(img_id)[0]['file_name']
        path = path.split('_0.jpg')[0]

        # print(f'path: {path}, anno: {annotations}')

        with h5py.File(os.path.join(self.root_dir, path), 'r') as file:
            rgb_input = file['rgb_input'][:]

        # data augmentation: synthetize with noise
        is_synnoise = True
        syn_prob = 0.5
        syn_factor = random.uniform(0.1, 0.4)
        if is_synnoise and np.random.rand() > syn_prob:
            noise_df = pd.read_csv('/home/chun/SeaFOAM_chun/Mask_RCNN/noise.csv')
            selected_path = noise_df.iloc[random.randint(0, len(noise_df) - 1)]['File Path']
            with h5py.File(selected_path, 'r') as file_noise:
                rgb_input = rgb_input + file_noise['rgb_input'][:] * syn_factor
            rgb_input[:, :, 0] = normalize(rgb_input[:, :, 0])
            rgb_input[:, :, 1] = normalize(rgb_input[:, :, 1])
            rgb_input[:, :, 2] = normalize(rgb_input[:, :, 2])
        image = torch.from_numpy(rgb_input)
        image = image.permute(2, 1, 0)

        # resize image
        original_size = image.shape[1:]
        target_height, target_width = self.target_size

        # data augmentation: stretching
        is_resize = False
        if is_resize:
            resize_h_factor = random.uniform(0.8, 1.1)
            resize_w_factor = random.uniform(0.6, 1.1)
            target_height = int(target_height * resize_h_factor)
            target_width = int(target_width * resize_w_factor)

        image = F.resize(image, (target_height, target_width))

        # data augmentation: masking
        is_masking = False
        mask_prob = 0.5
        if is_masking and np.random.rand() > mask_prob:
            min_ratio, max_ratio = 0.1, 0.3
            _, H, W = image.shape
            num_mask_rows = int(H * random.uniform(min_ratio, max_ratio))
            masked_rows = random.sample(range(H), num_mask_rows)
            image[:, masked_rows, :] = 0.0
        
        # Load labels and bounding boxes
        boxes = []
        labels = []
        masks = []
        attention_masks = []
        areas = []
        iscrowd = []

        for ann in annotations:
            # Calculate the scale factors
            scale_x = target_width / original_size[1]
            scale_y = target_height / original_size[0]

            bbox = ann['bbox']
            boxes.append([bbox[0] * scale_x, bbox[1] * scale_y, (bbox[0] + bbox[2]) * scale_x, (bbox[1] + bbox[3]) * scale_y])
            labels.append(ann['category_id'])

            area = bbox[2] * bbox[3] * scale_x * scale_y
            areas.append(area)

            iscrowd.append(ann.get('iscrowd', 0))

            height, width = target_height, target_width
            mask = np.zeros((height, width))

            # param for gauss distribution
            std_dev = 50 * scale_x
            amplitude = 1.0

            for iseg in range(len(ann['segmentation'])):
                data = ann['segmentation'][iseg]
                if len(data) < 10:
                    continue
                data = [(int(data[i] * scale_x), int(data[i+1] * scale_y)) for i in range(0, len(data), 2)]
                data = interpolate_line_segments_int(data)

                for x, y in data:
                    if y < height:
                        mask[y] += gaussian_line(x, std_dev, amplitude, width)
                # for i in range(height):
                #     max_value = np.max(mask[i, :])
                #     if max_value > amplitude:
                #         mask[i, :] /= (max_value / amplitude)
            row_max = np.max(mask, axis=1)
            mask_rows_to_normalize = row_max > amplitude
            scales = row_max[mask_rows_to_normalize] / amplitude
            mask[mask_rows_to_normalize, :] /= scales[:, np.newaxis]

            # name_idx = 0
            # plot_mask(mask, title=f'a{name_idx}')
            # name_idx += 1
            
            mask = (mask * 255).astype(np.uint8)
            masks.append(mask)

            # Check for attention_mask in annotation, if not present generate an empty mask
            if 'attention_mask' in ann:
                attention_mask = self._generate_attention_mask(ann['attention_mask'], (height, width), original_size)
            else:
                # Generate an all zeros attention mask if not provided
                attention_mask = np.zeros((height, width), dtype=np.uint8)

            attention_masks.append(attention_mask)

        # Vertical flip
        is_Vflip = True
        if random.random() > 0.5 and is_Vflip:
            # Flip image
            image = F.vflip(image)

            # Flip boxes
            height = image.shape[1]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]

            # Flip masks
            masks = torch.stack([torch.flip(torch.as_tensor(mask, dtype=torch.uint8), dims=[0]) for mask in masks])

            # confident area
            attention_masks = torch.stack([torch.flip(torch.as_tensor(mask, dtype=torch.uint8), dims=[0]) for mask in attention_masks])
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.stack([torch.tensor(mask, dtype=torch.uint8) for mask in masks])
            attention_masks = torch.stack([torch.tensor(attention_mask, dtype=torch.uint8) for attention_mask in attention_masks])
            
        # Convert everything into a torch.Tensor and move to the designated device
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'attention_masks': attention_masks,
            'image_id': img_id,
            'area': areas,
            'iscrowd': iscrowd,
        }
        
        return image, target

    def __len__(self):
        return len(self.ids)

    def _generate_attention_mask(self, attention_mask_rects, mask_shape, original_shape):
        attention_mask = np.zeros(mask_shape, dtype=np.uint8)

        original_height, original_width = original_shape
        target_height, target_width = mask_shape
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        for rect in attention_mask_rects:
            x_min, y_min, width, height = rect
            x_min = int(x_min * scale_x)
            y_min = int(y_min * scale_y)
            width = int(width * scale_x)
            height = int(height * scale_y)
            x_max, y_max = x_min + width, y_min + height
            attention_mask[y_min:y_max, x_min:x_max] = 1

        return attention_mask


import matplotlib.pyplot as plt

def plot_mask(mask, title='Mask', figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.imshow(mask, cmap='gray', aspect='auto', origin='upper')
    plt.colorbar(label='Intensity')
    plt.title(title)
    plt.xlabel('Width (x)')
    plt.ylabel('Height (y)')
    plt.tight_layout()
    plt.savefig(f'{title}.jpg')
    plt.close()
