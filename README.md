# DASNet

DASNet detects signals in Distributed Acoustic Sensing (DAS) data, such as whale
calls, earthquakes, ships and T waves, classifies them, and picks their arrival
times. It is a Mask R-CNN with a Swin-Tiny backbone.

## Installation

```bash
git clone https://github.com/AI4EPS/DASNet.git
cd DASNet
conda create -n dasnet python=3.11 -y
conda activate dasnet
pip install torch torchvision   # choose the build for your CUDA version at https://pytorch.org
pip install -r requirements.txt
```

## Quick start

1. Download the pretrained model (trained on Monterey Bay DAS data):

   ```bash
   wget https://github.com/AI4EPS/DASNet/releases/download/v1.0.1/dasnet_swin_t_monterey.pth
   ```

2. Download example data. More files are in the
   [monterey_bay dataset](https://huggingface.co/datasets/AI4EPS/quakeflow_das/tree/main/monterey_bay/data).

   ```bash
   mkdir -p data
   wget -P data https://huggingface.co/datasets/AI4EPS/quakeflow_das/resolve/main/monterey_bay/data/20231109T130910Z.h5
   ```

3. List the input files in a CSV with a `path` column:

   ```bash
   (echo path; ls data/*.h5) > input_files.csv
   ```

4. Run prediction:

   ```bash
   python predict.py \
     --checkpoint dasnet_swin_t_monterey.pth \
     --data-list input_files.csv \
     --output-dir results \
     --plot-figure
   ```

   Add `--device cpu` if you have no GPU.

## Input data

Each HDF5 file needs a `data` dataset with shape `(channels, time_samples)` in
strain rate, and a `dt_s` attribute with the sampling interval in seconds.

The pretrained model was trained on the Monterey Bay cable. For other cables,
train your own model (see [Training](#training)).

## Output

Prediction writes `results/<file>.json` for each input file:

```json
{
  "file_name": "20231109T130910Z.h5",
  "resize_scale": 0.5,
  "instances": [
    {
      "box": [465.4, 5418.1, 2840.6, 7173.9],
      "score": 0.9996,
      "label": 3,
      "picks_by_stream": [
        {"stream": 0, "picks": [[466.0, 7096.0], [468.0, 7094.0]], "pick_scores": [0.55, 0.60]}
      ]
    }
  ]
}
```

- `box`: `[channel_start, time_start, channel_end, time_end]`, in channel and time-sample indices of the input file.
- `picks`: `[channel, time_sample]` arrival picks. With the default `resize_scale` of 0.5, there is a pick on every second channel.
- `picks_by_stream`: earthquakes (label 8) have two streams, 0 = P and 1 = S. Other labels have one.
- `score`: detection confidence. Detections below `--min-probability` (default 0.3) are dropped.

Labels of the pretrained model:

| Label | Signal |
|---|---|
| 1 | Blue whale A |
| 2 | Blue whale B |
| 3 | Fin whale |
| 4 | Others / unknown |
| 5 | Blue whale D call |
| 6 | T wave |
| 7 | Ship |
| 8 | Earthquake (P and S) |
| 9 | Sei whale |

With `--plot-figure`, figures are saved in `results/figures/`. Their axes are in
the downsampled grid (input indices × `resize_scale`).

## Training

Training labels are a COCO JSON file. Each `images` entry points to an HDF5 file
through `file_name`, with `width` = number of time samples and `height` = number
of channels. Each annotation has:

- `bbox`: `[time_start, channel_start, time_width, channel_height]`
- `segmentation`: the arrival as a flattened `[time, channel, ...]` polyline
- `segmentation_aux`: a second arrival (e.g. S), for classes with two streams
- `category_id`: integer class ID

Example: one class plus background, with two arrival streams for class 1:

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
  --resize-scale 1.0 \
  --transform-min-size 2048 \
  --transform-max-size 4096 \
  --mask-streams-by-class 1:2 \
  --use-learned-temporal-crop \
  --rpn-anchor-auto \
  --batch-size 1 \
  --epochs 100
```

Set `--num-classes` to the number of classes + 1. For multiple GPUs, use
`torchrun --standalone --nproc_per_node=2 train.py ...`. The best model is saved
as `output/run_01/model_best.pth` and can be used with `predict.py --checkpoint`.
