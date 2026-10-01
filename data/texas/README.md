# Texas 48-hour storm benchmark

Inputs of the synthetic Texas benchmark: shallow-water simulations over a fixed 500 x 500 domain,
used to train and evaluate the Texas CLDNet and LDNet (`code/ldnet/`, `configs/{cldnet,ldnet}/texas.json`)
and the FNO and VAE–ConvLSTM baselines (`code/fno/`, `code/vae_convlstm/`).

> **This is not the Des Plaines / Illinois dataset.** The main release described in the
> top-level `README.md` — 94 Illinois storms, the April 2013 flood-of-record,
> and the 90 train / 3 shared validation/test split in `splits/` — is a different domain entirely. Nothing here maps onto
> that split, and the sample indices below are unrelated to it.

## Contents

**In this repository: inputs only.** The DEM (in `sample_00001`), the 120 hyetographs
(`rain_source.npy`), the per-storm metadata (`readme.txt`), and the rainfall mean/std used by the Texas
LDNet and CLDNet (`rain_mean.npy`, `rain_std.npy`). `vae_and_latents_texas/` holds the normalized rain arrays of the
VAE–ConvLSTM baseline, which fix the row order of the arrays it regenerates. The `flow_variables.npy` simulation
outputs (65 GB) are not included.

| | Train | Test |
|---|---|---|
| Samples | 100 (`sample_00001` – `sample_00100`) | 20 (`sample_00101` – `sample_00120`) |

```
data/texas/train_dataset/sample_00001/DEM.npy             (3, 500, 500)       float32   (sample_00001 only)
                                     /rain_source.npy     (49, 2)             float32
                                     /readme.txt          per-storm simulation metadata (JSON)
                         /sample_00002/...
data/texas/test_dataset/sample_00101/...
data/texas/rain_mean.npy, rain_std.npy                    rainfall standardization (Texas LDNet/CLDNet)
data/texas/vae_and_latents_texas/rain_{train,test}.npy    (100|20, 192, 1)    float32   (VAE-ConvLSTM, normalized)
data/texas/SHA256SUMS                                     checksums
(not included: flow_variables.npy, (193, 3, 500, 500) float32 per storm, the 15-min simulation output)
```

## Arrays

**`DEM.npy`** `(3, 500, 500)` — channel 0 is the x coordinate, channel 1 the y
coordinate (both cell centres in metres, 15 to 14985), channel 2 the bed elevation in
metres (189.4 to 334.1). **Byte-identical in all 120 samples**; the domain is fixed and
only the rainfall forcing varies. It is stored once, in `train_dataset/sample_00001/`.

**`flow_variables.npy`** `(193, 3, 500, 500)` — the reference shallow-water solution at
full 15-minute output resolution over 48 simulated hours. Channel 0 is water depth `h`
(m), channels 1 and 2 are unit discharge `hU_x` and `hU_y` (m²/s).

**`rain_source.npy`** `(49, 2)` — column 0 is time in seconds (0 to 172800, hourly),
column 1 the spatially uniform rainfall rate in m/s. One rain source over the whole
domain; `readme.txt` reports the same storm in mm/h.

**`readme.txt`** — JSON metadata written by the simulator: grid extent and cell size,
run time, initial and boundary conditions, rainfall summary, and Manning / infiltration
parameters.

## Simulation setup

Identical across all 120 storms:

- 500 x 500 cells at 30 m = 225 km²
- `run_time = [0, 172800, 900, 1800]` — 48 hours simulated, output every 900 s (15 min),
  giving 193 saved states
- Manning coefficient 0.035 everywhere; no sewer sink, no infiltration
- Dry start (`h0 = 0`, `hU0 = 0`); outline boundary is a fall condition with `h` and `hU`
  fixed at zero (1996 boundary cells)
- One spatially uniform rain source at 3600 s temporal resolution

Despite the source folder name `texas_24hrs_DEM7`, the simulations are **48 hours** long.

## Rainfall

| | Peak (mm/h) | Total (mm) | Mean (mm/h) |
|---|---|---|---|
| Train (100) | 11.8 – 251.3 | 96.4 – 294.3 | 2.01 – 6.13 |
| Test (20) | 43.9 – 305.4 | 357.8 (all identical) | 7.45 (all identical) |

### Half the storms are time-reversed copies

**The 120 samples are 60 distinct hyetographs, each stored twice.** Samples
`(2m-1, 2m)` are exact time-reversals of one another over the 24-hour rain window
(rows 1-24 of `rain_source.npy`): verified for all 60 pairs at a maximum difference of
**0.0**, i.e. bit-for-bit reversed. So the dataset holds

| | Samples | Distinct hyetographs |
|---|---|---|
| Train | 100 | **50** |
| Test | 20 | **10** |

Two things follow, and they pull in opposite directions:

- **This is not train/test leakage.** No pair straddles the boundary — pairs are
  `(1,2) … (99,100)` in train and `(101,102) … (119,120)` in test. A test storm's mirror
  image is never in training.
- **But the rainfall diversity is half the sample count.** The held-out set explores
  **10** temporal patterns, not 20, and the training set 50, not 100. Effective sample
  size for anything driven by rainfall shape is half of what the sample count suggests.

The *trajectories* are genuinely distinct — the shallow-water equations are not
time-reversible, so reversing the forcing yields a different flow field (max difference
32.9 between samples 1 and 2, 41.6 between 101 and 102). The 120 simulations are real
and separate; it is the forcing diversity that is halved.

**The test set is not a random hold-out.** All 20 test storms deliver exactly the same
total rainfall depth (357.78 mm) at the same mean intensity, and differ only in how that
depth is distributed in time (and, per the pairing above, in only 10 distinct ways).
They are also wetter than every training storm — the heaviest training total is 294 mm. Read held-out metrics
with that in mind: they measure generalization to new temporal rainfall patterns at a
fixed, out-of-range total depth, not to new storms drawn from the training distribution.

## Model-ready arrays

The Texas LDNet and CLDNet read `data/postprocessed/texas/` (regenerated, not included): for storm `k`,
`flow_variables_traj<k>.npy` holds the 48 hourly states (15-min indices 4, 8, ..., 192 of `flow_variables.npy`)
on all 250,000 cells, shape `(1, 48, 250000, 3)`, float16, and `rain_source_traj<k>.npy` holds rows 0-47 of
column 1 of `rain_source.npy` (m/s), shape `(1, 48, 1)`. Rainfall is standardized with `rain_mean.npy` and
`rain_std.npy` (this folder), computed over the 100 training storms.

