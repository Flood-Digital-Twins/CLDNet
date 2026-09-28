"""Helpers for the SynxFlow/HiPIMS event scripts.

Extracted verbatim from `utils.py` in the original simulation tree, reduced to the
functions the three deposited scripts import. Nothing here depends on anything outside
this folder.

Note: with the deposited configuration neither entry point actually executes —
`add_filling_period` runs only when `--filling_hours > 0` (0 in every deposited run) and
`coords_min_dem_in_circle` only when the module-level `shift_gauges` flag is True (it is
False). They are kept so the imports resolve and so the options remain usable.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as geom


def min_dem_in_circle(dem, lon, lat, radius_m):
    # 1. Ensure DEM is in EPSG:5070
    dem_5070 = dem.rio.reproject("EPSG:5070")

    # 2. Make point GeoDataFrame in WGS84
    gdf = gpd.GeoDataFrame(
        geometry=[geom.Point(lon, lat)],
        crs="EPSG:4326"
    )

    # 3. Reproject point to EPSG:5070
    gdf_5070 = gdf.to_crs("EPSG:5070")

    # 4. Make circular buffer in meters
    circle = gdf_5070.buffer(radius_m).iloc[0]

    # 5. Clip DEM to circle
    dem_clip = dem_5070.rio.clip([circle], dem_5070.rio.crs, drop=True)

    # 6. Return minimum value
    return float(dem_clip.min().values)


def coords_min_dem_in_circle(dem, coords, crs, radius_m):
    # 1. Ensure DEM is in EPSG:5070
    dem_5070 = dem.rio.reproject("EPSG:5070")
    # 2. Make point GeoDataFrame in WGS84
    if crs == 'EPSG:4326':
        gdf = gpd.GeoDataFrame(
            geometry=[geom.Point(coords[0], coords[1])],
            crs="EPSG:4326"
        )
    elif crs == 'EPSG:5070':
        x_proj = coords[0]
        y_proj = coords[1]
        gdf = gpd.GeoDataFrame(
            geometry=[geom.Point(x_proj, y_proj)],
            crs="EPSG:5070"
        )
    # 3. Reproject point to EPSG:5070
    gdf_5070 = gdf.to_crs("EPSG:5070")
    # 4. Make circular buffer in meters
    circle = gdf_5070.buffer(radius_m).iloc[0]
    # 5. Clip DEM to circle
    dem_clip = dem_5070.rio.clip([circle], dem_5070.rio.crs, drop=True)
    # 6. Return the coordinates of the minimum and its value
    dem_clip_2d = dem_clip.squeeze()  # removes band dimension if present
    arr = np.array(dem_clip_2d.values, dtype=float)
    iy, ix = np.unravel_index(np.nanargmin(arr), arr.shape)
    x_coord = float(dem_clip_2d.x[ix])
    y_coord = float(dem_clip_2d.y[iy])
    val_min = float(dem_clip_2d.values[iy, ix])
    return [x_coord, y_coord], val_min


def add_filling_period(df, filling_hours, filling_intensity_mm_hr, anchor_time=None):
    """
    Add uniform rainfall rows before an anchor time (default = first index).
    No duplicate timestamps.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex")
    df = df.sort_index()

    # Use the first timestamp as anchor if not given
    anchor = pd.to_datetime(anchor_time) if anchor_time is not None else df.index[0]

    # Determine hourly step (assume constant)
    if len(df.index) > 1:
        dt = df.index[1] - df.index[0]
    else:
        dt = pd.Timedelta(hours=1)

    # Build strictly earlier timestamps
    new_times = [anchor - dt * i for i in range(filling_hours, 0, -1)]

    filler = pd.DataFrame(
        filling_intensity_mm_hr,
        index=pd.DatetimeIndex(new_times),
        columns=df.columns,
    )

    out = pd.concat([filler, df])
    out = out[~out.index.duplicated(keep="last")]
    out.sort_index(inplace=True)
    return out
