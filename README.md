# DASNet

DASNet trains a modified Mask R-CNN model to detect DAS events and extract arrival picks
from local HDF5 files.

## Installation

Use a Python environment with a PyTorch and torchvision build compatible with
your CUDA driver, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Input data

Preprocessed HDF5 files must contain a `data` dataset with shape
`(number_of_channels, number_of_time_samples)` and a scalar `dt_s` attribute.
The data is expected to be strain rate.

Training annotations use COCO JSON. Each `images` entry must refer to an HDF5
file through `file_name`, with `width` set to the time-sample count and `height`
set to the channel count. An annotation uses:

- `bbox`: `[time_start, channel_start, time_width, channel_height]`
- `segmentation`: a flattened `[time, channel, ...]` polyline
- `category_id`: an integer class identifier

For multiple arrival phases, provide additional polyline fields such as
`segmentation_aux` and configure their count with `--mask-streams-by-class`.

## Training

The following command trains a one-class model plus background with two mask
streams. Adjust paths and geometry for your dataset. Change --num-classes to n+1 for n-class model.

```bash
python train.py \
  --data-path /path/to/train_h5 \
  --anno-path /path/to/train.json \
  --val-data-path /path/to/validation_h5 \
  --val-anno-path /path/to/validation.json \
  --output-dir output/run_01 \
  --num-classes 2 \
  --image-channels 1 \
  --use-crop-dataset \
  --crop-channels 2048 \
  --crop-samples 4096 \
  --transform-min-size 2048 \
  --transform-max-size 4096 \
  --mask-streams-by-class 1:2 \
  --use-learned-temporal-crop \
  --rpn-anchor-auto \
  --batch-size 1 \
  --epochs 100
```

For multi-GPU training, launch the same command through `torchrun`, for example
`torchrun --standalone --nproc_per_node=2 train.py ...`.

The output directory contains `checkpoint.pth` and the validation-loss-best
checkpoint `model_best.pth`.

## Prediction

Create a CSV with one local HDF5 path per row:

```csv
path
/path/to/input_001.h5
/path/to/input_002.h5
```

Run prediction with model options that exactly match the training run:

```bash
python predict.py \
  --checkpoint output/run_01/model_best.pth \
  --data-list input_files.csv \
  --output-dir results \
  --num-classes 2 \
  --image-channels 1 \
  --resize-scale 1.0 \
  --transform-min-size 2048 \
  --transform-max-size 4096 \
  --mask-streams-by-class 1:2 \
  --use-learned-temporal-crop \
  --rpn-anchor-auto \
  --rpn-anchor-samples 4096 \
  --rpn-anchor-channels 2048
```

For classic Monterey Bay data, omit `--use-learned-temporal-crop` unless the
checkpoint was trained with it, use `--image-channels 3`, and set the class,
transform, mask-branch, and RPN options to the checkpoint's training values.
For raw MBARI input, add `--from-raw --channel-start START --channel-end END`.

Prediction writes one JSON file per HDF5 input. Each record contains detection
boxes in `[channel_start, time_start, channel_end, time_end]` order and
per-stream pick coordinates in `[channel, time]` order.

The model architecture options, class count, input channel count, crop geometry,
mask-stream configuration, learned temporal crop option, and RPN anchor settings
must match the checkpoint. A mismatch can prevent checkpoint loading or produce
invalid predictions.
