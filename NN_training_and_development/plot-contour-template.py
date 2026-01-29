
# %%
# This scripts plots three contours from an ORCESTRA run for :
# 1) column-intg cloud liquid water
# 2) horizontal surface wind speed
# 3) surface precipitation rate
# 4) column-intg water vapor

# %%
# Setup environment

#%matplotlib inline

from pylab import *

import netCDF4 as nc
import xarray as xr
import numpy as np
import cartopy.feature as cf
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
#import matplotlib.font_managerquit()

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.feature as cfeature
import pandas as pd
import itertools
import os
import dask
import easygems.remap as egr
import datashader as ds
#import psutil
import seaborn
import matplotlib.colors as mcolors

from matplotlib import gridspec
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

os.environ['TEXMFHOME'] = '/home/m/m301067/texmf'
os.environ['PATH'] = '/sw/spack-levante/texlive-live2021-l5o6sw/bin/x86_64-linux:' + os.environ['PATH']
plt.rcParams.update({'font.size': 13, 'font.family': 'TimesNewRoman', 'text.usetex': True})
rcParams['axes.linewidth'] = 1.5
rcParams["axes.formatter.use_mathtext"]
    
# Customize the tick labels to remove the minus sign (its in °W)
def format_longitude(x, pos):
    return f"{abs(int(x))}"

# %%
# User-defined parameters

# Choose here the case to plot by prescribing the starting day (MMDD)
mmdd = "0825"


# Choose the timestep to plot (total of 289, once every 10 minutes)
time2plot = 258 #240=Day2:16h, 258=Day2:19h UTC

# Choose the resolution
ires = 1 #1:HD, 2:2K, 3:4k(~1.25km resolution)

# Update the figure's size according to target resolution
nx, ny = 1440 * ires, 720 * ires
dpi = 150*ires
plt.rcParams['figure.dpi'] = 150*ires #450: 4K, 300: 2K, 150: HD to stay coherent

# %%
# Reading the dataset

datadir = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/orcestra_1250m_"+mmdd+"/"
weightsdir = "/work/mh0492/m301067/orcestra/auxiliary-files/weights/"
datafile = "orcestra_1250m_"+mmdd+"_atm_2d_ml_DOM01_2024"+mmdd+ "T000000Z.nc"
meshdir = "/work/mh0492/m301067/orcestra/auxiliary-files/grids/"
meshname = "ORCESTRA_1250m_DOM01"

ds = xr.open_dataset(datadir+datafile, chunks={"ncells": -1})
grid = xr.open_dataset(meshdir+meshname+".nc")

lon, lat = np.meshgrid(
    np.linspace(grid.clon.min().values, grid.clon.max().values, nx), 
    np.linspace(grid.clat.min().values, grid.clat.max().values, ny)
)

# Compute (load) remapping weightsm
weightsname = "weights_"+meshname+"_ZoomLvl"+str(ires)
if os.path.exists(weightsdir+weightsname+".nc"):
    print("Reading weights")
    weights = xr.open_dataset(weightsdir+weightsname+".nc")
else:
    print("Calculating weights")
    weights = egr.compute_weights_delaunay((grid.clon, grid.clat), (lon.ravel(), lat.ravel()))
    weights.to_netcdf(weightsdir+weightsname+".nc", mode="w")

# Apply remapping function only to the variables we need
ds_remap = xr.apply_ufunc(
    egr.apply_weights,
    ds,
    kwargs=weights,
    input_core_dims=[["ncells"]],
    output_core_dims=[["xy"]],
    output_dtypes=["f4"],
    vectorize=True,
    dask="parallelized",
    dask_gufunc_kwargs={
        "output_sizes": {"xy": lon.size},
    },
)

# Assign the MultiIndex to the 'xy' dimension
ds_remap = ds_remap.assign_coords(
    xy=pd.MultiIndex.from_product(
        (np.degrees(lat[:, 0]), np.degrees(lon[0, :])), 
        names=("lat", "lon"),
    )
).unstack("xy")

# Extract and format the time dimension
times = ds_remap.time.values
num_snapshots = len(times)
print(f"Number of snapshots: {num_snapshots}, among which we plot number {time2plot}")
formatted_times = [pd.to_datetime(str(time)).strftime('%Y-%m-%dT%H-%M') for time in times]
formatted_time = formatted_times[time2plot]
    
# %% First example
# Plotting cloud liquid water path

cllvi = ds_remap.cllvi.isel(time=time2plot)

fig, ax = plt.subplots(figsize=(13, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plot = cllvi.plot(vmin=0, vmax=2, cmap=plt.cm.nipy_spectral, ax=ax, add_colorbar=False, transform=ccrs.PlateCarree())

# Customize the plot
ax.coastlines(color='white', linewidth=0.7)
ax.set_aspect('equal', adjustable='box')
xticks = np.linspace(-62, -10, 14)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
ax.set_yticks(np.linspace(-2, 22, 7), crs=ccrs.PlateCarree())
ax.set_xlim([-62, -10])
ax.set_ylim([-2, 22])
ax.set_xlabel('Longitude [°W]', color='white')
ax.set_ylabel('Latitude [°N]', color='white')
ax.grid(color='white', alpha=0.4, linestyle='dashed', linewidth=0.3)

# Set black background and white lines
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
for spine in ax.spines.values():
    spine.set_edgecolor('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Add colorbar
cax = fig.add_axes([ax.get_position().x1 + 0.006, ax.get_position().y0, 0.006, ax.get_position().height])
cbar = plt.colorbar(plot, cax=cax, ticks=np.linspace(0, 2, 11))
cbar.set_label('Cloud liquid path [kg/m$^2$]', fontsize=13, weight="bold", \
    rotation=270, labelpad=20, color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
cbar.outline.set_edgecolor('white')

# Add title with formatted timestamp
ax.set_title(formatted_time, color='white', fontsize=15)

# Display the figure
plt.show()

# Save the figure in the daily directory
#filename = ...
#plt.savefig(filename, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())

# %% Second example
# Plotting horizontal surface wind speed

# Compute hsws for the current timestep
uas = ds_remap.uas.isel(time=time2plot)
vas = ds_remap.vas.isel(time=time2plot)
hsws = np.sqrt(uas**2 + vas**2).compute()

fig, ax = plt.subplots(figsize=(13, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plot = hsws.plot(vmin=0, vmax=20, cmap=plt.cm.magma, ax=ax, add_colorbar=False, transform=ccrs.PlateCarree())

# Customize the plot
ax.coastlines(color='white', linewidth=0.7)
ax.set_aspect('equal', adjustable='box')
xticks = np.linspace(-62, -10, 14)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
ax.set_yticks(np.linspace(-2, 22, 7), crs=ccrs.PlateCarree())
ax.set_xlim([-62, -10])
ax.set_ylim([-2, 22])
ax.set_xlabel('Longitude [°W]', color='white')
ax.set_ylabel('Latitude [°N]', color='white')
ax.grid(color='white', alpha=0.4, linestyle='dashed', linewidth=0.3)

# Set black background and white lines
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
for spine in ax.spines.values():
    spine.set_edgecolor('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Add colorbar
cax = fig.add_axes([ax.get_position().x1 + 0.006, ax.get_position().y0, 0.006, ax.get_position().height])
cbar = plt.colorbar(plot, cax=cax, ticks=np.linspace(0, 20, 5))
cbar.set_label('Horizontal surface wind speed [m/s]', fontsize=13, weight="bold", \
    rotation=270, labelpad=20, color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
cbar.outline.set_edgecolor('white')

# Add title with formatted timestamp
ax.set_title(formatted_time, color='white', fontsize=15)

# Display the figure
plt.show()

# Save the figure in the daily directory
#filename = ...
#plt.savefig(filename, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())

# %% Third example
# Plotting surface precipitation flux

pr = ds_remap.pr.isel(time=time2plot).compute() * 3600
pr = xr.where(pr == 0, 1e-6, pr)
pr = np.log10(pr)

fig, ax = plt.subplots(figsize=(13, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plot = pr.plot(vmin=-1, vmax=2.5, cmap=plt.cm.CMRmap, ax=ax, add_colorbar=False, transform=ccrs.PlateCarree())

# Customize the plot
ax.coastlines(color='white', linewidth=0.7)
ax.set_aspect('equal', adjustable='box')
xticks = np.linspace(-62, -10, 14)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
ax.set_yticks(np.linspace(-2, 22, 7), crs=ccrs.PlateCarree())
ax.set_xlim([-62, -10])
ax.set_ylim([-2, 22])
ax.set_xlabel('Longitude [°W]', color='white')
ax.set_ylabel('Latitude [°N]', color='white')
ax.grid(color='white', alpha=0.4, linestyle='dashed', linewidth=0.3)

# Set black background and white lines
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
for spine in ax.spines.values():
    spine.set_edgecolor('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Add colorbar
cax = fig.add_axes([ax.get_position().x1 + 0.006, ax.get_position().y0, 0.006, ax.get_position().height])
cbar = plt.colorbar(plot, cax=cax, ticks=np.linspace(-1, 2.5, 8))
cbar.set_label('Precipitation flux (log10-base) [mm/h]', fontsize=13, weight="bold", \
    rotation=270, labelpad=20, color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
cbar.outline.set_edgecolor('white')

# Add title with formatted timestamp
ax.set_title(formatted_time, color='white', fontsize=15)

# Display the figure
plt.show()

# Save the figure in the daily directory
#filename = ...
#plt.savefig(filename, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())

# %% Fourth example
# Plotting precipitable water

prw = ds_remap.prw.isel(time=time2plot)

fig, ax = plt.subplots(figsize=(13, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plot = prw.plot(vmin=30, vmax=80, cmap=plt.cm.bone, ax=ax, add_colorbar=False, transform=ccrs.PlateCarree())

# Customize the plot
ax.coastlines(color='white', linewidth=0.7)
ax.set_aspect('equal', adjustable='box')
xticks = np.linspace(-62, -10, 14)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
ax.set_yticks(np.linspace(-2, 22, 7), crs=ccrs.PlateCarree())
ax.set_xlim([-62, -10])
ax.set_ylim([-2, 22])
ax.set_xlabel('Longitude [°W]', color='white')
ax.set_ylabel('Latitude [°N]', color='white')
ax.grid(color='white', alpha=0.4, linestyle='dashed', linewidth=0.3)
    
# Add isocontour at level 48
data_smoothed = gaussian_filter(ds_remap.prw.isel(time=time2plot).compute(), sigma=6)
contour = ax.contour(np.degrees(lon), np.degrees(lat), data_smoothed, levels=[48], \
    colors='orange', alpha=0.25)

# Set black background and white lines
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
for spine in ax.spines.values():
    spine.set_edgecolor('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Add colorbar
cax = fig.add_axes([ax.get_position().x1 + 0.006, ax.get_position().y0, 0.006, ax.get_position().height])
cbar = plt.colorbar(plot, cax=cax, ticks=np.linspace(30, 80, 6))
cbar.set_label('Precipitable water path [kg/m$^2$]', fontsize=13, weight="bold", \
    rotation=270, labelpad=20, color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
cbar.outline.set_edgecolor('white')

# Add title with formatted timestamp
ax.set_title(formatted_time, color='white', fontsize=15)

# Display the figure
plt.show()

# Save the figure in the daily directory
#filename = ...
#plt.savefig(filename, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())

