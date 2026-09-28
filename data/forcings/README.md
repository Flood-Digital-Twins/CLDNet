# Rainfall forcing — Des Plaines / Illinois (HUC8 07120004)

Stage IV hourly precipitation driving the 30 m shallow-water simulations. The simulated
trajectories are not included in this repository; see the top-level README to regenerate them.

**94 storms: 93 from the numbered catalog plus the 2013 flood-of-record.**

## Layout

```
forcings/
├── event_0/  event_1/  …  event_109/       93 numbered storms
│     ├── event_<id>.nc                     simulator INPUT
│     └── rain_source.npy                   simulator output, the model input
├── event_2013-04-17_2013-04-21/            the 2013 flood-of-record
└── SHA256SUMS                              covers every file
```

Each storm folder holds two files (the manifest lives at this level rather than inside
them):

| File | Contents |
|---|---|
| `event_<id>.nc` | the Stage IV extract the simulator **reads**: `prcp(time, y, x)` = `(97, 39, 13)` float64 mm/hr, EPSG:5070, 4 km cells, with real timestamps and coordinates |
| `rain_source.npy` | `(97, 507)` float64, **mm/hr** — 97 hourly steps × 507 rain cells |

The `.nc` is the input and `rain_source.npy` an output of the same run. The 507 columns are
the `.nc` grid flattened row-major as `(y, x)`: `prcp.reshape(97, -1)` reproduces
`rain_source.npy` to within **1.4e-14**, the round-trip error of writing and re-reading
`rain_source.csv` as text. The `.nc` is therefore the higher-precision copy, and the one
to use when re-running the simulator.

`rain_source.csv`, a labelled text copy that the simulation script writes alongside, is not included.
The 507 cells are a 39 × 13 Stage IV grid at 4 km, flattened row-major as `(y, x)`.

Because the `.nc` files are here, `preprocessing/simulation/run_hipims_event.py` can be
re-run from the release to regenerate any storm's trajectory; it reads the `.nc`, not the
`rain_source` files.

Numbered folders take the event index from the Stage IV catalog. The flood-of-record has
no index there, so it is named for the window it actually covers.

## How it was produced

`rain_source.csv` is written by `preprocessing/simulation/run_hipims_event.py` from the per-event Stage IV file
`data_for_HiPIMS_30/precipitation_st4_midwest_fill_nan_adjust/event_<id>.nc`;
`collect_training_data.py` then subsamples it to `rain_source.npy`. The command behind
each numbered storm:

```bash
python preprocessing/simulation/run_hipims_event.py      --event_idx <id> --device_id <0|1>
python preprocessing/simulation/collect_training_data.py --event_idx <id> --device_id <0|1>
```

`event_2013-04-17_2013-04-21` came from `preprocessing/simulation/run_hipims_event_for_validation.py`, whose
arguments are set in the script rather than passed on the command line.


## Caveats

- **20 duplicate storms were dropped.** The catalog contains 20 pairs carrying
  byte-identical rainfall — same values, same grid origin, same timestamps. Only the
  lower index of each pair is deposited. Dropped → kept:
  85→25, 86→26, 87→27, 88→28, 89→29, 95→35, 96→36, 97→37, 98→38, 99→39,
  110→50, 111→51, 112→52, 113→53, 114→54, 115→55, 116→56, 117→57, 118→58, 119→59.
  The duplication originates in the Stage IV catalog, upstream of all simulation code.
  The model-ready arrays (`data/postprocessed/illinois/`, regenerated) carry the same 94 events, so their IDs
  line up with this folder — with one exception: the flood-of-record is
  `event_2013-04-17_2013-04-21` here and `traj120` there. Match the 93 numbered storms
  by id, and that one by name.

- **Seven indices never existed**: 7, 17, 44, 60, 66, 67, 72. So 0–119 yields 113 files,
  and 113 − 20 duplicates = 93 numbered storms.

- **The flood-of-record folder name rounds to whole days.** The run covers
  **2013-04-17 07:00 → 2013-04-21 07:00** (97 hourly steps), so it starts and ends at
  07:00 rather than midnight. Its source folder is named `event_2013-04-10_2013-04-23_…`
  and a 336-hour file for that 13-day span exists upstream, but the 97-hour extract is
  what was simulated — the source folder name overstates the period.

- **`event_13` and `event_84` overlap the flood-of-record.** They cover 68% and 93% of
  its hours, and essentially all of its rainfall falls inside their windows. They are
  *not* duplicates — cell-by-cell correlation is 0.36 and 0.38 against a −0.07 baseline
  for unrelated storms, because each extraction uses its own grid origin. Same weather,
  different sampling. Relevant to whether the flood-of-record validation is independent
  of training.

- **Rain cells are mapped to the DEM by index, not by coordinates.** `run_hipims_event.py`
  upscales the 39 × 13 grid onto the DEM with `np.kron`, ignoring the `x`/`y` coordinates
  in the `.nc`. Each event's Stage IV extraction has its own origin (36 distinct origins
  across the catalog, differing by up to ~3 km), so column *j* does not refer to the same
  ground in every storm, even though it always drives the same DEM cells.

- Values are float64 as written by the simulator; no rounding or unit conversion was
  applied. The m/s conversion HiPIMS consumes is applied in memory at run time and is
  not reflected in these files.
