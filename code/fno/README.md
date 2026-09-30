# FNO baseline (fixed-DEM Texas case)

Autoregressive Fourier Neural Operator surrogate used as the FNO comparison in the
paper. The model maps

```
(DEM_x, DEM_y, elevation, slope_x, slope_y, rainfall, h, hU_x, hU_y)_t  ->  (h, hU_x, hU_y)_{t+1}
```

on a fixed 500 x 500, 30 m grid at a 1-hour model time step, and is rolled forward
from the initial condition alone at inference time.

> **Dataset note.** This baseline was trained on a **separate Texas domain**, not on
> the Des Plaines / Illinois dataset described in the top-level `README.md`. Its
> 100 training / 20 held-out trajectories are unrelated to the Des Plaines split in
> `splits/`. The "texas" in the file names is accurate, not a leftover.

## Layout

```
code/fno/
  fno_autoregressive_training_texas_fixed_DEM.py   training + end-of-run rollout
  fno_autoregressive_inference_texas_fixed_DEM.py  rollout + relative RMSE / RMSE / R^2
  fno_utils/                                        self-contained helpers
  requirements.txt
configs/fno/run_config.json                         config of the deposited run
checkpoints/fno/checkpoint_epoch_50.pth             deposited weights
checkpoints/fno/fno_epoch_loss.npy                  training loss per epoch (50, 1)
```

`fno_utils/` has no imports outside this directory, so `code/fno/` runs on its own.
Both scripts resolve every default path from their own location — nothing depends on
an absolute path. Run them from any working directory:

```bash
cd code/fno
python fno_autoregressive_inference_texas_fixed_DEM.py --checkpoint-path ../../checkpoints/fno
```

Run artifacts land in `code/fno/outputs/` (git-ignored); `--results-directory` /
`--results-root` redirect them.

## Environment

```bash
pip install -r requirements.txt
```

Python >= 3.11. The deposited checkpoint was produced with Python 3.13.5,
torch 2.7.1+cu126 and neuraloperator 1.0.2.

## Data layout

The dataset is deposited under `data/texas/`, one directory per trajectory. The public
GitHub repository holds the inputs only (the DEM in `sample_00001` and every
`rain_source.npy`); the `flow_variables.npy` simulations (65 GB) are not included, and
training and scoring need them.

```
data/texas/train_dataset/sample_00001/DEM.npy             (3, 500, 500)       x, y, elevation
                                     /flow_variables.npy  (193, 3, 500, 500)  h, hU_x, hU_y every 15 min
                                     /rain_source.npy     (49, 2)             time, rainfall (m/s)
                                     /readme.txt          grid, rainfall and Manning metadata
                                     /flow_variables.gif  preview animation
                         /sample_00002/...
data/texas/test_dataset/sample_00101/...
```

All arrays are `float32`, which is what the scripts cast to on load. `readme.txt`
records the per-storm simulation metadata (grid extent, rainfall totals, Manning
coefficient, boundary conditions).

`flow_variables.npy` keeps the full 15-min output; the scripts subsample it with
`TEMPORAL_STRIDE = 4` down to the 49 hourly states the model steps through. The DEM
is identical across every sample and is read once from the first one. Training uses
samples 1-100; evaluation adds the held-out samples 101-120.

Both scripts default to these paths, so no flags are needed. `--dataset-directory`,
`--train-dataset-directory` and `--test-dataset-directory` override them.

## Reproducing the deposited checkpoint

```bash
python fno_autoregressive_training_texas_fixed_DEM.py \
  --dataset-directory <path>/train_dataset \
  --fourier-modes 32 32 \
  --hidden-channels 64 \
  --prediction-time-horizon 10 \
  --batch-size 2 \
  --num-epochs 50 \
  --seed 42
```

Everything else stays at its default: `--lr 1e-3`, `--weight-decay 1e-9`, AdamW,
mean MSE loss, 4 FNO layers, `--cell-size 30.0`, `--save-every 5`, samples 1-100.
8,965,059 parameters. The original run used `--device cuda:3`, which is why the run
directory was named `device_cuda3_modes_32x32_hidden_64_horizon_10_batch_2`; the
device only affects that directory name.

Rainfall is standardized with statistics computed over the training set and stored in
the checkpoint, so inference reuses them rather than recomputing on the evaluation set:

```
rain_source_mean = 1.0423133289805264e-06
rain_source_std  = 3.306995040475158e-06
```

## Scoring the checkpoint

```bash
python fno_autoregressive_inference_texas_fixed_DEM.py \
  --checkpoint-path ../../checkpoints/fno/checkpoint_epoch_50.pth \
  --train-dataset-directory <path>/train_dataset \
  --test-dataset-directory <path>/test_dataset \
  --batch-size 2
```

The model architecture comes from `configs/fno/run_config.json` (found automatically;
override with `--run-config`). The script rolls each trajectory forward 47 steps from
its initial state and writes predictions, metrics, GIFs and bar charts per split.

Metrics from the original inference run of this checkpoint, over the full 100-sample
training set and 20-sample held-out set:

| Split | Metric | Water depth | Discharge-x | Discharge-y | All |
|---|---|---|---|---|---|
| train (100) | relative RMSE | 0.1196 | 0.2170 | 0.1947 | 0.1738 |
| train (100) | RMSE | 0.0354 | 0.0495 | 0.0504 | 0.0456 |
| train (100) | R² | 0.9855 | 0.9529 | 0.9621 | 0.9697 |
| test (20) | relative RMSE | 0.1278 | 0.2151 | 0.1965 | 0.1810 |
| test (20) | RMSE | 0.0470 | 0.0706 | 0.0712 | 0.0639 |
| test (20) | R² | 0.9834 | 0.9537 | 0.9614 | 0.9671 |

Note that `autoregressive_predictions.npy` is large: 2.9 GB for the 20 test
trajectories, 14.7 GB for the 100 training ones.

## Notes

- The deposited checkpoint is a slimmed copy of the original
  `checkpoint_epoch_50.pth`: AdamW optimizer state and 50 embedded PNG diagnostic
  frames were dropped, taking it from 301 MB to 71.5 MB. Every model weight is
  byte-identical; `normalization_stats`, `epoch` and `epoch_loss_list` are intact.
  It cannot be used to *resume* training, only to evaluate.
- The training script still embeds those diagnostic frames in the checkpoints it
  writes, which is why in-flight checkpoints grow over epochs.
- Water depth is clamped to be non-negative after every forward pass
  (`postprocess_prediction`); the discharge channels are left unconstrained.
- The hyperparameter sweep behind the deposited run covered
  modes in {8, 16, 32}², hidden channels in {32, 64, 128} and rollout horizon in
  {1, 3, 5, 10}.
