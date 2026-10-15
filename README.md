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

New training defaults to **Swin-Tiny** with ImageNet initialization. Use
`--backbone resnet50` to select the legacy backbone, or
`--backbone-weights none` for random initialization. Swin's four stages feed the
existing 128-channel FPN; their outputs are pooled by two to retain the legacy
FPN strides. All Swin stages are trainable.

The box head and fixed **42 × 42 mask RoIAlign** remain unchanged. This branch
does not include the experimental high-resolution decoder, attention, magnitude,
or polarity heads.

The following command trains a one-class model plus background with two mask
streams. Adjust paths and geometry for your dataset. Set `--num-classes` to
`n + 1` for an `n`-class model.

```bash
python train.py \
  --backbone swin_t \
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
  --resize-scale 1.0 \
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

Prediction restores architecture and input geometry from the training
checkpoint, without downloading initialization weights:

```bash
python predict.py \
  --checkpoint output/run_01/model_best.pth \
  --data-list input_files.csv \
  --output-dir results
```

Checkpoints predating backbone metadata are treated as ResNet50. Explicit
conflicting backbone options are rejected. For checkpoints without training
arguments, supply the class count and matching input/mask/RPN settings manually.
For raw MBARI input, add `--from-raw --channel-start START --channel-end END`.

Prediction writes one JSON file per HDF5 input. Each record contains detection
boxes in `[channel_start, time_start, channel_end, time_end]` order and
per-stream pick coordinates in `[channel, time]` order.

With `--plot-figure`, all detected classes keep their boxes and labels. Arrival
masks and picks are drawn only for Monterey categories 2 (Blue whale B), 3
(Fin whale), and 8 (earthquake P/S). Use `--plot-arrival-classes` to supply other
category IDs for a different dataset. This display filter does not change JSON
predictions. Box labels use black text on a translucent white background and
stay inside the plotting area.

The model architecture options, class count, input channel count, crop geometry,
mask-stream configuration, learned temporal crop option, and RPN anchor settings
must match the checkpoint. A mismatch can prevent checkpoint loading or produce
invalid predictions.

## Monterey temporal-crop geometry

The training and prediction defaults are `--resize-scale 0.5`,
`--transform-min-size 1422`, and `--transform-max-size 6000`.
For native-resolution crop datasets, specify the desired resize and transform
geometry explicitly (for example `--resize-scale 1.0` with 2048/4096).

With `--use-learned-temporal-crop`, each stream uses a window of length
`max(1, alpha * proposal_channel_width)` around its predicted anchor. The
window keeps original image coordinates, including negative/beyond-end time
coordinates. Feature pooling, target projection and mask pasting use the same
window; no proposal-dependent global shift or feature expansion is applied.

This corrects the historical image-width window and boundary-shift behavior.
Existing checkpoint tensor shapes remain compatible, but their inference
geometry changes; reevaluate predictions and retrain for controlled comparisons.

Run geometry, backbone, backward-pass and checkpoint regression tests with:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```
