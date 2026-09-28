# SynxFlow / HiPIMS simulation scripts — Des Plaines / Illinois

The three scripts that generated the Illinois dataset: they build a HiPIMS case from the
static terrain and a storm's Stage IV forcing, run the shallow-water solver, and reduce the
raw solver output to arrays.

```
preprocessing/simulation/
├── run_hipims_event.py                 one numbered storm, end to end
├── run_hipims_event_for_validation.py  the 2013 flood-of-record
├── collect_training_data.py            solver output -> flow_variables/DEM/rain_source .npy
└── sim_utils.py                        the two helpers the above import
```

## What produced the deposited data

```bash
cd <repository-root>

# 93 numbered storms (device 0 and 1 were alternated across two shells)
python preprocessing/simulation/run_hipims_event.py      --event_idx <id> --device_id <0|1>
python preprocessing/simulation/collect_training_data.py --event_idx <id> --device_id <0|1>

# the flood-of-record; its settings are hard-coded in the script, not passed as flags
python preprocessing/simulation/run_hipims_event_for_validation.py
python preprocessing/simulation/collect_training_data.py --event_idx 2013-04-17_2013-04-21
```

`run_hipims_event.py` writes the case folder, `rain_source.csv`, `simulation_config.json`,
`readme.txt` and the raw `output/*.asc`. `collect_training_data.py` then stacks those into
`flow_variables.npy` `(97, 3, 5075, 1661)`, builds `DEM.npy`, subsamples `rain_source.npy`,
**and deletes `output/`** — which is why only a handful of raw solver outputs survive.

## Inputs and outputs

All defaults resolve from `Path(__file__).resolve().parents[2]`, i.e. the repository root,
so the scripts run from any working directory.

| Default | Path | Override |
|---|---|---|
| Static terrain, land cover, initial condition, gauges | `data/static/` | `--static-root` |
| Stage IV forcing for storm `<id>` | `data/forcings/event_<id>/event_<id>.nc` | `--forcing-root` |
| Case folder written | `data/sims_30/event_<id>_filling_hours_0_filling_intensity_mm_hr_0/` | `--out-root` |

## What was verified

`run_hipims_event.py --event_idx 0` was executed from an unrelated working directory with
the solver call stubbed out, writing to a scratch folder. It resolved all seven deposited
inputs and produced the complete 26-file case (573 MB). Comparing that case against the
original 2025 run of storm 0:

| | Result |
|---|---|
| `rain_source.csv` vs the deposited `data/forcings/event_0/rain_source.csv` | **identical** |
| `input/` tree (19 files: mesh, field, boundary, gauges) | **18 identical**, 1 differs — see below |
| `readme.txt` | identical values; formatting differs only as `np.float64(3769.956)` vs `3769.956`, a NumPy 2.x repr change |

**Not verified:** the solver run itself. SynxFlow is unmodified third-party code and a full
97-hour solve writes roughly 78 GB of `output/*.asc` per storm, so it was not re-run here.
What is established is that the deposited scripts hand the solver byte-identical inputs.

## Caveats

- **`device_setup.dat` disagrees with the recorded config in the original data.** Storm 0's
  `simulation_config.json` says `device_id: 0`, but the `input/device_setup.dat` in that
  case folder says `3`. The regenerated case writes `0`, matching the config. The GPU index
  affects nothing about the physics, but the deposited config is not a faithful record of
  which device ran.
- **`--watershed` is gone.** The originals took a single `--watershed` pointing at a
  `watersheds/huc8_07120004/` tree that does not exist in this release; it is replaced by
  `--static-root`, `--forcing-root` and `--out-root`. Everything else — `--event_idx`,
  `--device_id`, `--resolution`, the Manning and filling options — is unchanged.
- **`sim_utils.py` is a reduction of a 924-line `utils.py`** down to the two functions these
  scripts import, plus one helper. Neither runs under the deposited settings:
  `add_filling_period` needs `--filling_hours > 0` (always 0), and `coords_min_dem_in_circle`
  needs the module-level `shift_gauges` flag to be True (it is False). They are kept so the
  imports resolve. `add_filling_period` was checked to behave identically to the original.
- **`burned_dem=1` will not work.** `dem_burned_5070.tif` is not deposited, because every
  run used `burned_dem=0`.
- **The scripts start from a wet initial state**, loaded unconditionally from
  `data/static/initial_condition/`. `--filling_strategy` and `--initial_waterdepth` are
  inert: the branch that would use them is commented out. See `data/static/README.md`.
- Requires `synxflow` with GPU support, plus numpy, pandas, xarray, rasterio, rioxarray,
  geopandas, shapely, pyproj and matplotlib. Tested with the `flood_synxflow` environment
  (Python 3.10, NumPy 2.2.6).
