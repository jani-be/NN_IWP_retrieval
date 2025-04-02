"""
Colocating Simulation points and HALO flight

timewise
and
spatially

 
fist for one flight


#Output:
#Index of simulation points
# Needed Output:
#time steps
#-> needed for simulation

#Later 
#which simulation point refer to which time step
#-> for plotting
"""

#%% Loading packages
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import fsspec
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
#%% reading halo data lon,lat and time

DATE ="0825"

fs = fsspec.filesystem("ipns")
file_altitude = f'ipns://latest.orcestra-campaign.org/products/HALO/position_attitude/HALO-2024{DATE}a.zarr'
ds_halo = xr.open_dataset(file_altitude,engine ="zarr")
ds_halo = ds_halo.drop_vars(["heading","pitch","roll"])
# dropping halo < 10000 m
ds_halo = ds_halo.where((ds_halo.alt >= 10000),drop =True)

#%% reading sim data lon,lat
meshdir = "/work/mh0492/m301067/orcestra/auxiliary-files/grids/"
meshname = "ORCESTRA_1250m_DOM01"
grid = xr.open_dataset(meshdir+meshname+".nc")
grid = grid.drop_dims(["vertex","edge","no","nc","nv","ne","max_chdom","cell_grf","edge_grf","vert_grf"])





#%% choosing spatial comparison method
#inspo by Max
#First for every halo point
#then setting a minimal difference
#hence reducing data by factor 30
#or 
#directly choosing every 30th values

#noch kein Automatismus
#temporally
day = (DATE[2:5])
month = DATE[0:2]
t_periods = 5
t_steps =xr.date_range("2024-"+month+"-"+day+" 9:45:00",periods=t_periods,freq="2h")

time_intervals=np.concatenate([xr.date_range(t,freq="1min",periods =30)for t in t_steps])
#time_intervals=np.concatenate([xr.date_range(t,freq="2s",periods =900)for t in t_steps])
# hier halo samplen? außerdem mehr als 15 min verbieten

ds_halo=ds_halo.sel(time=time_intervals,method="nearest")

def find_nn(lon_model, lat_model, lon_point, lat_point):

    lon_model = np.deg2rad(lon_model)
    lat_model = np.deg2rad(lat_model)
    lon_point = np.deg2rad(lon_point)
    lat_point = np.deg2rad(lat_point)

    lon_point_arr = np.zeros(len(lon_model))
    lat_point_arr = np.zeros(len(lat_model))
    lon_point_arr = np.full(len(lon_model),lon_point)
    lat_point_arr = np.full(len(lat_model),lat_point)


    dlon = lon_point_arr - lon_model
    dlat = lat_point_arr - lat_model 

    # calculate 2D distances between model and 
       # point on the sphere of earth using the harvesine function
    a = np.sin(dlat/2)**2 + np.cos(lat_model) * np.cos(float(lat_point)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    d = c * r

    # returning model grid index of closest grid box 
    # to given point coordinate and the corresponding distance
    nn_ind = np.unravel_index(np.argmin(d, axis=None), d.shape)
    d_min = d[nn_ind[0]]

    return nn_ind[0], d_min



lon_min=ds_halo.lon.min()
lat_min=ds_halo.lat.min()
lon_max=ds_halo.lon.max()
lat_max=ds_halo.lat.max()
grid = grid.where(
    (grid.clon>=np.deg2rad(lon_min))&(grid.clon<=np.deg2rad(lon_max))&
    (grid.clat>=np.deg2rad(lat_min))&(grid.clat<=np.deg2rad(lat_max)),
    drop=True)
ICON_lat = np.rad2deg(grid.clat.to_pandas())
ICON_lon=np.rad2deg(grid.clon.to_pandas())
# find index of nearest ICON grid cell to HALO coordinates
#nn_index, colloc_dists = find_nn(
#    ICON_lon,ICON_lat,ds_halo.lon,ds_halo.lat)
nn_d=[]
[nn_d.append(find_nn(ICON_lon,ICON_lat,lon,lat)) for lon, lat in zip(ds_halo.lon, ds_halo.lat)] 
####

np.save("/home/u/u301032/orcestra/NN_IWP_retrieval/NN_training_and_development/cells_0824.npy",np.array(nn_d))
#%% Choosing time reduction
#maximum 30 min?


#%% Output:
#Index of simulation points
#time steps
#-> needed for simulation

#Later 
#which simulation point refer to which time step
#-> for plotting
