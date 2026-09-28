# Watershed preparation — DEM, burned DEM, land cover, gauges

`watershed_data_preparation.ipynb` builds every static geospatial input for the Des Plaines
watershed (HUC8 **07120004**) at 30 m, straight from public web services.

It covers what is split across `preprocessing/dem/` and `preprocessing/manning_nlcd/`;
those folders hold pointers back here rather than a second copy.

## What it produces

All written to `watersheds/huc8_07120004/data_for_HiPIMS_30/`, relative to the working
directory the notebook is run from:

| Output | Deposited as | Used by the simulator? |
|---|---|---|
| `dem_5070.tif` | `data/static/dem_5070.tif` | **yes** — terrain and rain-mask template |
| `landcover_5070.tif` | `data/static/landcover_5070.tif` | source for the binary Manning field |
| `gages.geojson` | not deposited | superseded by `gage_index_mapping_with_lat_lon.csv` |
| `dem_burned_5070.tif` | not deposited | no — every run used `burned_dem=0` |
| `dem_4326.tif`, `dem_burned_4326.tif`, `landcover_4326.tif` | not deposited | no — never read by any run |
| `mask_largest_water_body.npy` | not deposited | no — referenced only in commented-out code |

`landcover_5070_processed.tif` (the binary water/land raster the runs actually use) is **not**
made here — `run_hipims_event.py` derives it on first use as `landcover == 11 or 12 → 1, else 0`.

## How it works

| Step | Cells | What happens |
|---|---|---|
| Watershed boundary | 3–8 | `pynhd`/`pygeohydro` WBD lookup for HUC8 `07120004`; `gagesii` for the USGS gauges |
| DEM | 11–13 | `py3dep.get_dem(...)` at 30 m, native **EPSG:5070**, plus a 4326 reprojection |
| Hydrological conditioning | 14–16 | `pysheds` flow direction, pit filling and inflation |
| DEM burning | 19–23 | the largest waterbody is burned into the conditioned DEM → `dem_burned_5070.tif` |
| Land cover | 24–25, 28 | `pygeohydro.nlcd_bygeom(..., years={"cover": [2021]})`, reprojected to match the DEM |

Key parameters, all set in the notebook: `resolution = 30`, `watershed_id = "07120004"`,
NLCD cover year **2021**, gauge record window `2010-09-24` to `2025-09-24`.

## Caveats

- **It cannot be re-run offline or deterministically.** Every input is fetched live from
  3DEP, the Watershed Boundary Dataset, NLCD and USGS NWIS. Re-running it later may return
  different data if those services revise their products. The rasters in `data/static/` are
  the authoritative copies of what the paper actually used.
- **Paths are relative to the working directory**, not to the notebook. Run it from a
  directory where `watersheds/huc8_07120004/data_for_HiPIMS_30/` is the intended destination.
  It creates that tree itself with `os.makedirs(..., exist_ok=True)`.
- **Outputs were cleared before deposit** (33.9 MB → 38 kB; the embedded maps and figures were
  the entire difference). Every code cell is byte-identical to the original — verified, not
  assumed. The original, with outputs, is kept by the authors.
- **The notebook is broader than this watershed.** `watershed_id` also carries commented-out
  HUC6/HUC4/HUC2 examples, and two different `gage_indices` lists appear; the values above are
  the ones left active.
- **Extra packages** beyond the simulation environment: `py3dep`, `pygeohydro`, `pynhd`,
  `pygeoutils`, `pysheds`, `geopandas`, `rioxarray`, `rasterio`, `shapely`, `pyproj`,
  `requests`, `requests_cache`, `datashader`, `mpld3`, `dill`, `scipy`, `xarray`.
