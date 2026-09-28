# %%
import os 
import sys
from pathlib import Path
# Resolve every default from this file's location so the script runs from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# %%
#%%  Import the required libraries
from synxflow import IO
import os
start_dir = os.getcwd()
from synxflow.IO.demo_functions import get_sample_data
import pandas as pd
import matplotlib.pyplot as plt
import sys
import json
import numpy as np
import rasterio
import rioxarray
import time
import argparse
from synxflow import flood
import xarray as xr
from pyproj import Transformer
import geopandas as gpd
import shapely.geometry as geom
crs_projector = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
from sim_utils import coords_min_dem_in_circle, add_filling_period
import shutil

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--ngpus", type=int, default=1)
parser.add_argument("--static-root", type=str, default=str(REPO_ROOT / "data" / "static"),
                    help="DEM and land cover")
parser.add_argument("--out-root", type=str, default=str(REPO_ROOT / "data" / "sims_30"),
                    help="case folders written by run_hipims_event.py")
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
DEM = IO.Raster(str(STATIC / 'dem_5070.tif')) # load the file into a Raster object

# %%
DEM_array = DEM.array

# %%
DEM.mapshow()

# %%
plt.imshow(DEM_array, cmap='viridis')

# %%
cell_size = 30.0
nx = DEM_array.shape[1]
ny = DEM_array.shape[0]
x_min = 15.0
y_min = 15.0
x_coords = x_min + np.arange(nx) * cell_size
y_coords = y_min + np.arange(ny) * cell_size
X_coords, Y_coords = np.meshgrid(x_coords, y_coords)
Y_coords = np.flipud(Y_coords)  # Flip Y_coords to match the DEM orientation

# %%
# For X_coords
im1 = plt.imshow(X_coords, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()))
plt.colorbar(im1, label='X coordinate')  # Add colorbar with label
plt.title('X Coordinates')
plt.show()

# For Y_coords
im2 = plt.imshow(Y_coords, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()))
plt.colorbar(im2, label='Y coordinate')  # Add colorbar with label
plt.title('Y Coordinates')
plt.show()

# %%
case_folder = parent_path

# %%
whole_DEM = np.stack([X_coords, Y_coords, DEM_array], axis=0)
np.save(os.path.join(case_folder, 'DEM.npy'), whole_DEM)

# %%
rain_source = pd.read_csv(os.path.join(case_folder, 'rain_source.csv'))

# %%
# rain_source

# %%
# for i in range(1, rain_source.shape[1] - 2+1):
#     plt.plot(rain_source[str(i)].values)
#     plt.title(F'precipitation (time index: 0 hr - {rain_source.shape[0] - 1} hr)')

# %%
selected_start_hours = 0
selected_end_hours = 96

# %%
# for i in range(1, rain_source.shape[1] - 2+1):
#     plt.plot(rain_source[str(i)].values[selected_start_hours:selected_end_hours+1])
#     plt.title(f'precipitation (time index: {selected_start_hours} hr - {selected_end_hours} hr)')

# %%
interested_time_indices = [i * 3600 for i in range(selected_start_hours, selected_end_hours + 1)]

# %%
print('saving flow variables as npy file ...')

flow_variables = []
for time_index in interested_time_indices:
    h = np.loadtxt(os.path.join(case_folder, 'output', f'h_{time_index}.asc'), skiprows=6)
    hUx = np.loadtxt(os.path.join(case_folder, 'output', f'hUx_{time_index}.asc'), skiprows=6)
    hUy = np.loadtxt(os.path.join(case_folder, 'output', f'hUy_{time_index}.asc'), skiprows=6)
    flow_variables.append(np.stack([h, hUx, hUy], axis=0))
flow_variables = np.array(flow_variables)
flow_variables[flow_variables == -9999.0] = np.nan

np.save(os.path.join(case_folder, 'flow_variables.npy'), flow_variables)

# %%
# old_flow_variables = []
# for time_index in interested_time_indices:
#     h = np.loadtxt(os.path.join(case_folder, 'output', f'old_h_{time_index}.asc'), skiprows=6)
#     hUx = np.loadtxt(os.path.join(case_folder, 'output', f'old_hUx_{time_index}.asc'), skiprows=6)
#     hUy = np.loadtxt(os.path.join(case_folder, 'output', f'old_hUy_{time_index}.asc'), skiprows=6)
#     old_flow_variables.append(np.stack([h, hUx, hUy], axis=0))
# old_flow_variables = np.array(old_flow_variables)
# old_flow_variables[old_flow_variables == -9999.0] = np.nan

# np.save(os.path.join(case_folder, 'old_flow_variables.npy'), old_flow_variables)

# %%
# new_flow_variables = []
# for time_index in interested_time_indices:
#     h = np.loadtxt(os.path.join(case_folder, 'output', f'new_h_{time_index}.asc'), skiprows=6)
#     hUx = np.loadtxt(os.path.join(case_folder, 'output', f'new_hUx_{time_index}.asc'), skiprows=6)
#     hUy = np.loadtxt(os.path.join(case_folder, 'output', f'new_hUy_{time_index}.asc'), skiprows=6)
#     new_flow_variables.append(np.stack([h, hUx, hUy], axis=0))
# new_flow_variables = np.array(new_flow_variables)
# new_flow_variables[new_flow_variables == -9999.0] = np.nan

# np.save(os.path.join(case_folder, 'new_flow_variables.npy'), new_flow_variables)

# %%
# cmap_dict = { 
#     'water_depth': 'Blues', 
#     'discharge_x': 'RdBu_r',
#     'discharge_y': 'RdBu_r'
# }

# clip_percentiles = (0.1, 99.9)

# vmin_dict = {
#     'water_depth': None,
#     'discharge_x': None,
#     'discharge_y': None
# }

# vmax_dict = {
#     'water_depth': None,
#     'discharge_x': None,
#     'discharge_y': None
# }


# real_values_min = {
#     'water_depth': None, 
#     'discharge_x': None,
#     'discharge_y': None
# }

# real_values_max = {
#     'water_depth': None,
#     'discharge_x': None,    
#     'discharge_y': None
# }

# vmin_dict['water_depth'], vmax_dict['water_depth'] = np.nanpercentile(flow_variables[:, 0, :, :], clip_percentiles)

# # --- discharge_x ---
# px_low, px_high = np.nanpercentile(flow_variables[:, 1, :, :], clip_percentiles)
# A = max(abs(px_low), abs(px_high))
# vmin_dict['discharge_x'] = -A
# vmax_dict['discharge_x'] =  A

# # --- discharge_y ---
# py_low, py_high = np.nanpercentile(flow_variables[:, 2, :, :], clip_percentiles)
# B = max(abs(py_low), abs(py_high))
# vmin_dict['discharge_y'] = -B
# vmax_dict['discharge_y'] =  B


# real_values_min['water_depth'] = np.nanmin(flow_variables[:, 0, :, :])
# real_values_max['water_depth'] = np.nanmax(flow_variables[:, 0, :, :])
# real_values_min['discharge_x'] = np.nanmin(flow_variables[:, 1, :, :])
# real_values_max['discharge_x'] = np.nanmax(flow_variables[:, 1, :, :])
# real_values_min['discharge_y'] = np.nanmin(flow_variables[:, 2, :, :])
# real_values_max['discharge_y'] = np.nanmax(flow_variables[:, 2, :, :])

# # %%
# print(f'vmin_dict {(vmin_dict)}')
# print(f'vmax_dict {(vmax_dict)}')
# print(f'real_values_min {(real_values_min)}')
# print(f'real_values_max {(real_values_max)}')

# # %%
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import io
# import imageio

# # %%
# frames = []

# for i in range(flow_variables.shape[0]):
#     fig, axes = plt.subplots(1, 3, figsize=(24, 8))

#     for ax, var, cmap, vmin, vmax, title in zip(
#         axes,
#         flow_variables[i],
#         [cmap_dict['water_depth'], cmap_dict['discharge_x'], cmap_dict['discharge_y']],
#         [vmin_dict['water_depth'], vmin_dict['discharge_x'], vmin_dict['discharge_y']],
#         [vmax_dict['water_depth'], vmax_dict['discharge_x'], vmax_dict['discharge_y']],
#         ['Water depth', 'Discharge-X', 'Discharge-Y']
#     ):
#         im = ax.imshow(var, cmap=cmap, vmin=vmin, vmax=vmax)
#         ax.set_title(title, fontsize=20)
        
#         # Increase tick label size
#         ax.tick_params(axis='both', which='major', labelsize=16)
#         ax.tick_params(axis='both', which='minor', labelsize=16)

#         # Create colorbar
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        
#         # Increase colorbar tick label size
#         cbar.ax.tick_params(labelsize=16)

#     # Reduce horizontal spacing between plots
#     fig.subplots_adjust(wspace=-0.5)  # closer spacing

#     # Global title
#     fig.suptitle(
#         f"t = {i} hours", 
#         fontsize=25, 
#         x=0.5, 
#         y=1.02, 
#         ha='center'
#     )

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
#     buf.seek(0)
#     frames.append(imageio.v2.imread(buf))
#     buf.close()
#     plt.close(fig)

# imageio.mimsave(os.path.join(case_folder, "flow_variables.gif"), frames, duration=1.0, loop=0)


# # %%
# water_depth = flow_variables[:, 0, :, :].flatten()
# discharge_x = flow_variables[:, 1, :, :].flatten()
# discharge_y = flow_variables[:, 2, :, :].flatten()

# non_nan_water_depth = water_depth[~np.isnan(water_depth)]
# non_nan_discharge_x = discharge_x[~np.isnan(discharge_x)]
# non_nan_discharge_y = discharge_y[~np.isnan(discharge_y)]

# # %%
# threshold = 1e-1  # choose your threshold

# filtered_water_depth = non_nan_water_depth[np.abs(non_nan_water_depth) > threshold]
# filtered_discharge_x = non_nan_discharge_x[np.abs(non_nan_discharge_x) > threshold]
# filtered_discharge_y = non_nan_discharge_y[np.abs(non_nan_discharge_y) > threshold]

# # %%
# plt.hist(filtered_water_depth, bins=100)
# plt.title(f'Histogram of Water Depth (|value| > {threshold})')
# plt.xlabel('Water Depth')
# plt.ylabel('Frequency')
# plt.show()

# plt.hist(filtered_discharge_x, bins=100)
# plt.title(f'Histogram of Discharge X (|value| > {threshold})')
# plt.xlabel('Discharge X')
# plt.ylabel('Frequency')
# plt.show()

# plt.hist(filtered_discharge_y, bins=100)
# plt.title(f'Histogram of Discharge Y (|value| > {threshold})')
# plt.xlabel('Discharge Y')
# plt.ylabel('Frequency')
# plt.show()

# %%
subsample_rain_source = rain_source[rain_source['0'].isin(interested_time_indices)].values[:, 2:]
subsample_rain_source = subsample_rain_source.astype(float)
np.save(os.path.join(case_folder, 'rain_source.npy'), subsample_rain_source)

# %%
# old_dt_list = []
# new_dt_list = []
# for time_index in interested_time_indices:
#     old_dt = np.loadtxt(os.path.join(case_folder, 'output', f'old_dt_{time_index}.txt'))
#     new_dt = np.loadtxt(os.path.join(case_folder, 'output', f'new_dt_{time_index}.txt'))
#     old_dt_list.append(float(old_dt))
#     new_dt_list.append(float(new_dt))
# old_dt_array = np.array(old_dt_list)
# new_dt_array = np.array(new_dt_list)
# np.save(os.path.join(case_folder, 'old_dt.npy'), old_dt_array)
# np.save(os.path.join(case_folder, 'new_dt.npy'), new_dt_array)

# %%
if os.path.exists(os.path.join(parent_path, 'output')):
    shutil.rmtree(os.path.join(parent_path, 'output'))
# %%



