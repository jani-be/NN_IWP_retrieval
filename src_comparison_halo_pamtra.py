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

#os.environ['TEXMFHOME'] = '/home/m/m301067/texmf'
#os.environ['PATH'] = '/sw/spack-levante/texlive-live2021-l5o6sw/bin/x86_64-linux:' + os.environ['PATH']
#plt.rcParams.update({'font.size': 13, 'font.family': 'TimesNewRoman', 'text.usetex': True})
#rcParams['axes.linewidth'] = 1.5
#rcParams["axes.formatter.use_mathtext"]
    
# Customize the tick labels to remove the minus sign (its in °W)
def format_longitude(x, pos):
    return f"{abs(int(x))}"

# %%
def remap(ds,ires=3,input_core_dim="ncells"):
    """
    Uses the weights given by sim for remapping
    :param str input_core_dim: for frac land use "cell"
    :param int ires: resolution of remapping. can be a value of 1, 2 or 3. 3 has highest resolution: 1 #1:HD, 2:2K, 3:4k(~1.25km resolution)
    """

    # Update the figure's size according to target resolution
    nx, ny = 1440 * ires, 720 * ires
    dpi = 150*ires
    plt.rcParams['figure.dpi'] = 150*ires #450: 4K, 300: 2K, 150: HD to stay coherent


    # Reading the dataset

    #datadir = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/orcestra_1250m_"+mmdd+"/"
    weightsdir = "/work/mh0492/m301067/orcestra/auxiliary-files/weights/"
    #datafile = "orcestra_1250m_"+mmdd+"_atm_2d_ml_DOM01_2024"+mmdd+ "T000000Z.nc"
    meshdir = "/work/mh0492/m301067/orcestra/auxiliary-files/grids/"
    meshname = "ORCESTRA_1250m_DOM01"

    #ds = xr.open_dataset(datadir+datafile, chunks={"ncells": -1})
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
        input_core_dims=[[input_core_dim]],
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


    return ds_remap

def plotting_R(ds,time2plot):
        # Extract and format the time dimension
    times = ds.time.values
    num_snapshots = len(times)
    print(f"Number of snapshots: {num_snapshots}, among which we plot number {time2plot}")
    formatted_times = [pd.to_datetime(str(time)).strftime('%Y-%m-%dT%H-%M') for time in times]
    formatted_time = formatted_times[time2plot]
    return num_snapshots, formatted_time


def read_grid_and_cell_data():
  meshdir = "/work/mh0492/m301067/orcestra/auxiliary-files/grids/"
  meshname = "ORCESTRA_1250m_DOM01"
  DATE="0829"
  appendix = ""
  path_sim = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/"
  path = path_sim + f"orcestra_1250m_{DATE+appendix}/"
  frac_land_file= path + "bc_land_frac.nc"
  grid = xr.open_dataset(meshdir+meshname+".nc",chunks="auto")
  frac_land= xr.open_dataset(frac_land_file,chunks="auto")
  return grid,frac_land

def cut_data_to_area_of_interest(ds,s_lat = -2,w_lon = -62,e_lon= -16,n_lat=22,division_factor=1000): #Koordinaten eigentlich in Dictionary
  # probably outdated
  #lonlat Grenzen übernehmen
  grid,frac_land = read_grid_and_cell_data()
  lat = np.rad2deg(grid.clat.to_pandas()[frac_land.sea.to_pandas()==1])
  lon=np.rad2deg(grid.clon.to_pandas()[frac_land.sea.to_pandas()==1])
  lat=lat[(lat <= n_lat )&(lat >= s_lat)]
  lon=lon[(lon <= e_lon)&(lon >= w_lon)]
  common_idx=lon.index.intersection(lat.index)
  idx_list=common_idx.to_list()
  #Reducing data size by factor 1000
  common_idx=idx_list[0::division_factor]
  ds=ds.sel(ncells=common_idx)

  frac_land = frac_land.drop_dims("nv")
  frac_land = frac_land.sel(cell = common_idx)

  #oder über selection?
  #ds.sel()
  #TODO plot 1000. Icon selection, make sure index passt, und wird übergeben für Vergleich für gridded notwendig
  return ds,common_idx,lon,lat

def cut_to_area(ds,s_lat = -2,w_lon = -62,e_lon= -16,n_lat=22):
  """
  cuts area of dataset to smaller size  
  :param float res: resolution of longitude and latitude 
  """
  ds=ds.sel(lat=slice(s_lat,n_lat),lon=slice(w_lon,e_lon))
  
  return ds

def coarse_for_pamtra(ds,s_lat = -2,w_lon = -62,e_lon= -16,n_lat=22,res=1):
  """
  coarsens dataset (and cuts it to area)
  :param float res: resolution of longitude and latitude 
  """
  ds=cut_to_area(ds,s_lat,w_lon,e_lon,n_lat)

  lat_range=np.arange(s_lat,n_lat+0.49*res,res)
  lon_range=np.arange(w_lon,e_lon+0.49*res,res)  
  ds=ds.sel(lat=lat_range,lon=lon_range,method="nearest")
  return ds