# %%
from synxflow import IO
import os
from pathlib import Path
start_dir = os.getcwd()          # synxflow's solver chdir()s; we restore this afterwards

# Every default below is resolved from this file's location, so the script runs from any
# working directory.  preprocessing/simulation/ -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Make sim_utils importable however this file is invoked (python script.py, runpy,
# or import) -- not only when the interpreter happens to seed sys.path with our dir.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from synxflow.IO.demo_functions import get_sample_data
import pandas as pd
import matplotlib.pyplot as plt
import sys
import json
import numpy as np
import rasterio
import rioxarray
from sim_utils import add_filling_period, coords_min_dem_in_circle
import time
import argparse
from synxflow import flood
import xarray as xr
from pyproj import Transformer
import geopandas as gpd
import shapely.geometry as geom
crs_projector = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--ngpus", type=int, default=1)
parser.add_argument("--static-root", type=str, default=str(REPO_ROOT / "data" / "static"),
                    help="DEM, land cover, initial condition and gauge positions")
parser.add_argument("--forcing-root", type=str, default=str(REPO_ROOT / "data" / "forcings"),
                    help="per-event Stage IV .nc, as data/forcings/event_<id>/event_<id>.nc")
parser.add_argument("--out-root", type=str, default=str(REPO_ROOT / "data" / "sims_30"),
                    help="where the case folder is written")
parser.add_argument("--burned_dem", type=int, default=0)
parser.add_argument("--variable_manning", type=int, default=0)

parser.add_argument("--manning_coefficient_water", type=float, default=0.02)
parser.add_argument("--manning_coefficient_land", type=float, default=0.05)

parser.add_argument("--event_idx", type=str, default='0')

parser.add_argument("--time_interval_hours", type=int, default=1)
parser.add_argument("--time_backup_hours", type=int, default=120)
parser.add_argument("--resolution", type=int, default=30)
parser.add_argument("--filling_strategy", type=str, default='dry')  # Either from filling_case or dry or waterbody_fixed_depth 
parser.add_argument("--initial_waterdepth", type=float, default=0.0)
parser.add_argument("--filling_hours", type=int, default=0)
parser.add_argument("--filling_intensity_mm_hr", type=float, default=0)

parser.add_argument("--device_id", type=int, default=0)
args = parser.parse_args()

# %%
# from argparse import Namespace
# args = Namespace(
#     ngpus=1,
#     watershed="watersheds/huc8_07120004/",
#     burned_dem=0,
#     variable_manning=0,
#     manning_coefficient_water=0.02,
#     manning_coefficient_land=0.05,
#     event_idx='0',
#     time_interval_hours=1,
#     time_backup_hours=120,
#     resolution=30,
#     filling_strategy='dry',
#     initial_waterdepth=0.0,
#     filling_hours=0,
#     filling_intensity_mm_hr=0,
#     device_id=0
# )

ngpus = args.ngpus
STATIC = Path(args.static_root)
FORCING = Path(args.forcing_root)
OUT_ROOT = Path(args.out_root)
burned_dem = args.burned_dem
variable_manning = args.variable_manning
manning_coefficient_water = args.manning_coefficient_water
manning_coefficient_land = args.manning_coefficient_land
event_idx = args.event_idx
time_interval = args.time_interval_hours * 3600  # in seconds
time_backup = args.time_backup_hours * 3600  # in seconds
resolution = args.resolution
filling_strategy = args.filling_strategy
initial_waterdepth = args.initial_waterdepth
filling_hours = args.filling_hours
filling_intensity_mm_hr = args.filling_intensity_mm_hr
device_id = args.device_id

# %%
case_name  = "event_" + event_idx + '_filling_hours_' +str(filling_hours) + '_filling_intensity_mm_hr_' +str(filling_intensity_mm_hr) + "/"   
parent_path = os.path.join(OUT_ROOT, case_name)
print(parent_path)
os.makedirs(parent_path, exist_ok=True)
figure_path = os.path.join(parent_path, 'figures')
os.makedirs(figure_path, exist_ok=True)

# %%
pr_ds = xr.open_dataset(os.path.join(FORCING, f"event_{event_idx}", f"event_{event_idx}.nc"))  # st4

pr = pr_ds["prcp"]
pr_flat = pr.stack(point=("y", "x"))
# Convert to DataFrame
df = pr_flat.to_pandas()

# print(df.head())
# print(df.info())
# print(list(df))

# Assign new column labels: 1..N (for spatial points)
df.columns = range(1, df.shape[1] + 1)

if filling_hours > 0.0:
    df = add_filling_period(df, filling_hours=filling_hours, filling_intensity_mm_hr=filling_intensity_mm_hr)

# Replace datetime index with numeric time in seconds
time_step_seconds = 3600
df.insert(0, 0, range(0, len(df) * time_step_seconds, time_step_seconds))
df.to_csv(os.path.join(parent_path, "rain_source.csv"), index=True)
rain_source_np = df.to_numpy()
rain_source_np[:,1:] = rain_source_np[:,1:]/(1000*3600) # Convert to m/sec

# %%
df

# %%
for i in range(1, rain_source_np.shape[1]):
    plt.plot(rain_source_np[:,0]/3600, rain_source_np[:,i])
    plt.xlabel('Time index (hour)')
    plt.ylabel('Rainfall rate (m/s)')
plt.title('Precipitation (time index: 0 hr - ' + str((rain_source_np.shape[0]-1)) + ' hr)')
plt.savefig(os.path.join(figure_path, 'rainfall_complete.png'))
plt.show()

# %%
start_time = 0
simulation_hours = int(len(pr_ds['time']))  + filling_hours
end_time = simulation_hours * 3600

# %%
if burned_dem == 1:
    print("Using burned DEM")
    DEM = IO.Raster(str(STATIC / 'dem_burned_5070.tif')) # load the file into a Raster object
    dem_rioxarray = rioxarray.open_rasterio(str(STATIC / 'dem_burned_5070.tif'))
else:
    print("Using unburned DEM")
    DEM = IO.Raster(str(STATIC / 'dem_5070.tif')) # load the file into a Raster object
    dem_rioxarray = rioxarray.open_rasterio(str(STATIC / 'dem_5070.tif'))

# %%
DEM.mapshow(figname = os.path.join(figure_path, 'DEM.png'))

# %%
# burned_DEM = IO.Raster(watershed + f'data_for_HiPIMS_{resolution}/dem_burned_5070.tif') # load the file into a Raster object
# burned_DEM.mapshow(figname = os.path.join(figure_path, 'burned_DEM.png'))
# plt.imshow(burned_DEM.array - DEM.array)
# plt.colorbar(label='meters')
# plt.title('Difference')
# plt.savefig(os.path.join(figure_path, 'DEM_difference.png'))
# plt.show()

# %%
if burned_dem == 1:
    rain_mask = IO.Raster(str(STATIC / 'dem_burned_5070.tif')) # load the file into a Raster object
else:
    rain_mask = IO.Raster(str(STATIC / 'dem_5070.tif')) # load the file into a Raster object
    
pr = pr_ds["prcp"]
ny_lr, nx_lr = pr.shape[1], pr.shape[2]
npoints = ny_lr * nx_lr
# Generate IDs: 0..N-1 in 2D grid
ids = np.arange(npoints).reshape(ny_lr, nx_lr)
# DEM size
ny_hr, nx_hr = rain_mask.array.shape
# Scale factors (ceil so we cover full DEM)
scale_y = int(np.ceil(ny_hr / ny_lr))
scale_x = int(np.ceil(nx_hr / nx_lr))
# Upscale IDs
ids_hr = np.kron(ids, np.ones((scale_y, scale_x), dtype=int))
# Now crop/pad to exact DEM shape
ids_hr = ids_hr[:ny_hr, :nx_hr]
# Assign to DEM raster
# rain_mask.array[:] = ids_hr[::-1,:] # [0]
rain_mask.array[:] = ids_hr[:,:]    #[1]

# %%
print("DEM shape:", rain_mask.array.shape)
print("Upscaled IDs shape:", ids_hr.shape)
print("IDs range:", ids_hr.min(), "to", ids_hr.max())
print("Rain mask shape:", rain_mask.array.shape)
print("IDs shape:", ids.shape)
rain_mask.mapshow(figname = os.path.join(figure_path, 'rain_mask.png'))

# %%
case_folder = parent_path
case_input = IO.InputModel(DEM, num_of_sections=ngpus, case_folder=case_folder)
#% We can then put some water in the catchment by setting an initial depth. In our case, the initial water depth is 0 m across the catchment.
################################################################################################################################################
#%% Set the initial condition
# Load the last time stamp of the filling simulaiton 


# if initial_waterdepth >0.0:
#     print('Using water mask')
#     water_mask_npy_path = watershed + f"data_for_HiPIMS_{resolution}/mask_largest_water_body.npy"
#     water_mask = np.load(water_mask_npy_path)
#     h0 = np.zeros_like(water_mask)
#     water_mask = water_mask.astype(float)
#     h0 = np.zeros_like(water_mask, dtype=float)
#     h0[water_mask > 0.5] = initial_waterdepth
#     case_input.set_initial_condition('h0', h0)

#     fig = plt.figure(figsize=(10, 6),dpi=250)
#     plt.imshow(water_mask, cmap='jet', vmin=0, vmax=0.5)
#     fig.savefig(os.path.join(figure_path,  'watermask.png'), dpi=300, bbox_inches='tight')
#     plt.close(fig)

# %%
h0 = np.loadtxt(STATIC / 'initial_condition' / 'h_0.asc.gz', skiprows=6)
hUx_0 = np.loadtxt(STATIC / 'initial_condition' / 'hUx_0.asc.gz', skiprows=6)
hUy_0 = np.loadtxt(STATIC / 'initial_condition' / 'hUy_0.asc.gz', skiprows=6)
case_input.set_initial_condition('h0', h0)
case_input.set_initial_condition('hU0x', hUx_0)
case_input.set_initial_condition('hU0y', hUy_0)

# %%
case_input.set_rainfall(rain_mask=rain_mask, rain_source=rain_source_np)

# %%
manning_coefficients_dict = {
    11: 0.04,   # Open Water
    12: 0.04,   # Perennial Ice/Snow
    21: 0.04,   # Developed, Open Space
    22: 0.10,   # Developed, Low Intensity
    23: 0.08,   # Developed, Medium Intensity
    24: 0.15,   # Developed, High Intensity
    31: 0.025,  # Rock/Sand/Clay
    41: 0.16,   # Deciduous Forest
    42: 0.16,   # Evergreen Forest
    43: 0.16,   # Mixed Forest
    51: 0.10,   # Dwarf Scrub
    52: 0.10,   # Shrub/Scrub
    71: 0.035,  # Grassland/Herbaceous
    72: 0.035,  # Sedge/Herbaceous
    73: 0.035,  # Lichens
    74: 0.035,  # Moss
    81: 0.03,   # Pasture/Hay
    82: 0.035,  # Cultivated Crops
    90: 0.12,   # Woody Wetlands
    95: 0.07    # Emergent Herbaceous Wetlands
}

binary_manning_coefficients_dict = {
    0: manning_coefficient_land,   # Land
    1: manning_coefficient_water  # Water
}

if variable_manning == 1:

    landcover = IO.Raster(str(STATIC / 'landcover_5070.tif')) # load the file into a Raster object
    case_input.set_landcover(landcover)
    landcover.mapshow(figname = os.path.join(figure_path, 'landcover.png'))
    landcover_classes = list(manning_coefficients_dict.keys())
    manning_values = list(manning_coefficients_dict.values())
    
    case_input.set_grid_parameter(manning={'param_value': manning_values,
                                            'land_value': landcover_classes,
                                            'default_value':manning_coefficient_water}
                                        )
    
elif variable_manning == 0:
    if not os.path.exists(STATIC / 'landcover_5070_processed.tif'):
        landcover = IO.Raster(str(STATIC / 'landcover_5070.tif')) # load the file into a Raster object

        landcover.array = np.where(
            np.isnan(landcover.array),
            np.nan,
            np.where((landcover.array == 11) | (landcover.array == 12), 1, 0)
        )

        output_file = str(STATIC / 'landcover_5070_processed.tif')
        landcover.write(output_file)

    landcover = IO.Raster(str(STATIC / 'landcover_5070_processed.tif')) # load the file into a Raster object
    case_input.set_landcover(landcover)
    case_input.set_grid_parameter(manning={'param_value': [manning_coefficient_land, manning_coefficient_water],
                                           'land_value': [0.0, 1.0],  # any 2 arbitrary numbers
                                           'default_value':manning_coefficient_land})

# %%
if variable_manning == 1:
    print("Using variable manning coefficients")
    indices_dict = {}
    for lc_class in manning_coefficients_dict.keys():
        indices = np.where(landcover.array == lc_class)
        indices_dict[lc_class] = indices
        print(f"Class {lc_class}: {len(indices[0])} cells")

else:
    print("Using binary manning coefficients")
    indices_dict = {}
    for lc_class in binary_manning_coefficients_dict.keys():
        indices = np.where(landcover.array == lc_class)
        indices_dict[lc_class] = indices
        print(f"Class {lc_class}: {len(indices[0])} cells")
print(" ")
print(f"total cells: {sum([len(v[0]) for v in indices_dict.values()])}")
print(f'nonnan cells in landcover map: {np.count_nonzero(~np.isnan(landcover.array))}')
print(f'nonnan cells in DEM: {np.count_nonzero(~np.isnan(DEM.array))}')
print("Using variable manning coefficients")

# %%
landcover.mapshow(figname = os.path.join(figure_path, 'landcover.png'))

# %%
# Get the Gauges_coords 
# Read your CSV
csv_path = str(STATIC / "gage_index_mapping_with_lat_lon.csv")
df = pd.read_csv(csv_path)
# Extract projected coordinates into numpy array
gauges_coords_5070 = df[["x_proj", "y_proj"]].to_numpy()
gauges_coords_4326 = df[["lon", "lat"]].to_numpy()

# print("Gauges coordinates (projected):", gauges_coords_5070)
# print("Gauges coordinates (geographic):", gauges_coords_4326)

#%%
# print(df.columns)

# %%
shift_gauges = False

if shift_gauges == True:
    radius_m = resolution*2
    coords = [658738.45765838, 2193615.75853093]
    coords_shifted, val_min= coords_min_dem_in_circle(dem_rioxarray,coords, "EPSG:5070", radius_m)

    coords = [-87.926466,42.489187]
    coords_shifted, val_min= coords_min_dem_in_circle(dem_rioxarray,coords, "EPSG:4326", radius_m)


#%% Loop to modify the gauges_coords_5070 based on the dem minimum in a circle
gauges_coords_5070_shifted = gauges_coords_5070.copy()
if shift_gauges == True:
    for gauge_idx , coords in enumerate(gauges_coords_5070):
        coords_shifted, val_min= coords_min_dem_in_circle(dem_rioxarray,coords, "EPSG:5070", radius_m)
        # gauges_coords_5070_shifted[gauge_idx, 0] = x_coord
        # gauges_coords_5070_shifted[gauge_idx, 1] = y_coord
        gauges_coords_5070_shifted[gauge_idx] = coords_shifted
        print(f"Gauge {gauge_idx}: Original coords: {coords}, Adjusted coords:{coords_shifted}, Min DEM value: {val_min:.2f}")

# %%
#%% Let's add the gauges# Lets add the gauges 
case_input.set_gauges_position(gauges_coords_5070_shifted)
case_input.set_runtime([start_time, end_time, time_interval, time_backup]) # [start time, end time, output interval, backup interval] in seconds
case_input.set_device_no(device_id)
case_input.write_input_files()


#%%
config = vars(args)
config.update({
    "simulation_hours": simulation_hours
})
save_path = os.path.join(parent_path +  "simulation_config.json")
with open(save_path, "w") as f:
    json.dump(config, f, indent=4)
print(f"✅ Configuration saved to: {save_path}")

# %%
tic = time.time()
if ngpus > 1:
    flood.run_mgpus(case_folder)
else:
    flood.run(case_folder)
toc = time.time()
print(f"Simulation runtime: {toc - tic:.2f} seconds")
os.chdir(start_dir)


config.update({
    "simulation_runtime_seconds": int(toc - tic)
})
with open(save_path, "w") as f:
    json.dump(config, f, indent=4)
print(f"✅ Configuration saved to: {save_path}")

# %%



