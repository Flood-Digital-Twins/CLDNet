# Manning field from NLCD

The NLCD land cover is downloaded in
**`preprocessing/watershed/watershed_data_preparation.ipynb`** (cells 24–25 and 28), which
also builds the DEM — see `preprocessing/watershed/README.md`.

The **Manning field itself is not a deposited raster.** It is derived at run time:

1. `preprocessing/simulation/run_hipims_event.py` reduces `landcover_5070.tif` to a binary
   water/land raster on first use — `landcover == 11 or 12 → 1, else 0` — and caches it as
   `landcover_5070_processed.tif` (regenerated automatically; not included).
2. Manning's *n* is then **0.02 over water and 0.05 over land**, set through
   `case_input.set_grid_parameter(...)`.

The 20-class NLCD coefficient table in `run_hipims_event.py` was **never applied**: every
deposited run used `variable_manning = 0`.

**Cell order of `manning.dat`.** SynxFlow writes the field to `input/field/manning.dat` with one value per valid
cell, numbered row by row from the *southern* row of the raster upward (the same order as `z.dat`). The dataset
scripts in `code/ldnet/` originally read it as if it started at the northern row, which mis-registered the Manning
channel of the Des Plaines terrain features; they now use the simulator's order (README, "Correction: Manning
channel").
