#%%
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import random
import sys
sys.path.append('/home/u/u301032/orcestra/NN_IWP_retrieval/')
import src_comparison_halo_pamtra as chp
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
#%%
# Choose Parameters #TODO In ein großes Dictionairy?
DATE= "0829" #"0927" #"0824" #
#NAME UNDER WHICH TO SAFE RUNS
output_name = "all_area_v1"
#Extend of Simulation area:
s_lat = -2
w_lon = -62
e_lon= -16
n_lat=22
flight_levels =[11400.0,11900.0,12650.0,13000.0,13250.0,13600.0,13900.0,14450.0,13900.0,15000.0	]
division_factor =1000 #1000 leads to ~ 6000 n_spatial

#%% reading file paths

if DATE == "0829":
    appendix = "-high3Drate"
else:
    appendix = "-rerun"

path_sim = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/"
path = path_sim + f"orcestra_1250m_{DATE+appendix}/"
height_file = path_sim + f"orcestra_1250m_0829-high3Drate/orcestra_1250m_0829-high3Drate_atm_vgrid_ml.nc"

meshdir = "/work/mh0492/m301067/orcestra/auxiliary-files/grids/"
meshname = "ORCESTRA_1250m_DOM01"
frac_land_file= path + "bc_land_frac.nc"

thermodyn_file = path + f"orcestra_1250m_{DATE+appendix}_atm_3d_thermodynamics_DOM01_2024{DATE}T000000Z.nc"
twodim_file = path + f"orcestra_1250m_{DATE+appendix}_atm_2d_ml_DOM01_2024{DATE}T000000Z.nc"

if DATE == "0829":
    hydrometeor1_file = path + f"orcestra_1250m_{DATE+appendix}_atm_3d_hydrometeors1_DOM01_2024{DATE}T000000Z.nc"
    hydrometeor2_file = path + f"orcestra_1250m_{DATE+appendix}_atm_3d_hydrometeors2_DOM01_2024{DATE}T000000Z.nc"
else:    
    hydrometeor_file = path + f"orcestra_1250m_{DATE+appendix}_atm_3d_hydrometeors_DOM01_2024{DATE}T000000Z.nc"


#%%
# reading files
grid = xr.open_dataset(meshdir+meshname+".nc",chunks="auto")[["clon","clat"]]
height = xr.open_dataset(height_file, chunks={"cell": -1})["zg"]
frac_land= xr.open_dataset(frac_land_file, chunks={"cell": -1})[['sea','land']]
#frac_land=frac_land.drop_dims("nv")
# %%%

# Limiting simulation data from 24h - 48 h
def _preprocess(x,  start_time="24h"):#, end_time="48h" ):
  if DATE == "0829":
      start_time = "12h"
  start=x.time[0].values+pd.Timedelta(start_time)
  #stop=x.time[0].values +pd.Timedelta(end_time) #last time step
  stop=x.time[-2].values #last time step
  t_steps =xr.date_range(start,stop,freq="2h")
  
  
  return x.sel(time=t_steps)

partial_func = partial(_preprocess,start_time="24h") #, end_time="48h")
twodim= _preprocess(xr.open_dataset(twodim_file, chunks={"ncells": -1}))
thermodyn =_preprocess(xr.open_dataset(thermodyn_file, chunks={"ncells": -1}))#xr.open_dataset(thermodyn_file)




if DATE == "0829":
    hyd1=  _preprocess(xr.open_dataset(hydrometeor1_file, chunks={"ncells": -1}))
    hyd2=  _preprocess(xr.open_dataset(hydrometeor2_file, chunks={"ncells": -1}))
    hyd=    xr.merge([hyd1,hyd2])
else:
    hyd = _preprocess(xr.open_dataset(hydrometeor_file, chunks={"ncells": -1}))

frac_land=chp.remap(frac_land,input_core_dim="cell")
frac_land=chp.coarse_for_pamtra(frac_land,res=0.25)

#for grid:

#grid = xr.open_dataset(meshdir+meshname+".nc",chunks="auto")[["clon","clat"]]

#grid2=grid.assign(cell_new=('cell',np.arange(12174205)))
grid = xr.open_dataset(f"/work/um0203/u301032/PAMTRA_input/prepped_grid2.nc",chunks={"cell": -1})

grid=chp.remap(grid2,input_core_dim="cell")
grid=chp.cut_to_area(grid)
grid=chp.coarse_for_pamtra(grid,res=0.25)
#get closest cell from rempaped data
lon=np.rad2deg(grid.clon.to_pandas())
lon_r=frac_land.coords["lon"].values
df_sort_lon = [ lon.iloc[(lon-input).abs().argsort()[:1]] for input in lon_r]
a=lon.iloc[(lon-lon_r[0]).abs().argsort()[:1]]
df=a
for i in df_sort_lon: 
    df=pd.concat([df,i])
df_lon=df.tail(-1)
lat=np.rad2deg(grid.clat.to_pandas())
lat_r=frac_land.coords["lat"].values
df_sort_lat = [ lat.iloc[(lat-input).abs().argsort()[:1]] for input in lat_r]
a=lat.iloc[(lat-lat_r[0]).abs().argsort()[:1]]
df=a
for i in df_sort_lat: 
    df=pd.concat([df,i])
df_lat=df.tail(-1)
# vgl elemente aus df_lat mit lat. ist es mehrfach enthalten?
df_lat.value_counts()
#%%
#haendische auswahl

lat = np.rad2deg(grid.clat.to_pandas()[frac_land.sea.to_pandas()==1])
lon=np.rad2deg(grid.clon.to_pandas()[frac_land.sea.to_pandas()==1])

try:    
    n_lat
    s_lat,
    e_lon,
    w_lon
except:
    print("no complete definiton of lon and lat area")    
else:
    lat=lat[(lat <= n_lat )&(lat >= s_lat)]
    lon=lon[(lon <= e_lon)&(lon >= w_lon)]
lon=lon[(lon <= e_lon)]
common_idx=lon.index.intersection(lat.index)
idx_list=common_idx.to_list()
df_lon=pd.DataFrame(data={'lon':lon})
df_lat=pd.DataFrame(data={'lat':lat})
df=pd.concat([df_lon,df_lat],axis=1,join='inner')
#%%
#grid,frac_land = chp.read_grid_and_cell_data()
def prep_ds(ds):
    ds=chp.remap(ds)
    ds=chp.cut_to_area(ds)
    ds=chp.coarse_for_pamtra(ds,res=0.25)
    ds.where((frac_land.sea==1).compute(),drop=True) 
    
    return ds

twodim=prep_ds(twodim[["ts","uas","vas"]])
twodim=twodim.drop_vars('height_2')
thermodyn=thermodyn[["pfull","ta"]]
thermodyn=prep_ds(thermodyn)#.drop_vars('height_bnds'))

height=prep_ds(height)
hyd = prep_ds(hyd[["qi","qs","qg","qc","qr","qv"]])#.drop_vars('height_bnds'))
ds=xr.merge([twodim,thermodyn,hyd,frac_land])
ds=ds.isel(height_2=0)

height # nur Höhe entnehmen. Kein remappen nötig
frac_land # still needs mask a
#ds.compute()
#ds.to_netcdf(f"/work/um0203/u301032/PAMTRA_input/prepped_Dataset_res025x025_2h_{DATE}.nc")
#frac_land.to_netcdf(f"/work/um0203/u301032/PAMTRA_input/prepped_frac_land_Dataset_res025x025_2h_{DATE}.nc")
# TODO get time
#height.to_netcdf(f"/work/um0203/u301032/PAMTRA_input/prepped_height_Dataset_res025x025_2h_{DATE}.nc")
#height.close()



variables2D_const = ["fract_land", 'lon', 'lat'] # 'topography_c' would be the ICON equivalent to the COSMO 'HSURF'
variables3D = ["ts"]#,"pres_sfc"]
variables4D_10m = ["uas","vas"]
variables4D = ["temp","pres","qv","qc","qi","qr","qs","qg","qh","qnc","qni","qnr","qns","qng","qnh"]

#%%

def _transpose_nc_var_by_dims(nc_var, dims):
  """Permutes netCDF4 variable dimensions according to the values given.

  This function makes sure that data is read in a well defined order of axes.
  An error is raised if the dims are not the same as used in the netcdf

  Parameters
  ----------
  nc_var : netCDF4._netCDF4.Variable
    netCDF Variable object
  dims : list of strings
    List with strings of dimensions.

  Returns
  -------
  var : np.ndarray
    An array with the data axes orderd according to dims.

  Raises
  ------
  ValueError

  """

  if len(dims) != len(set(dims)):
    raise ValueError('dims must not contain duplicates (%r).' % dims)
  if set(nc_var.dimensions) != set(dims):
    raise ValueError('Dimensions in nc_var (%r) are different from dims (%r).' % (
      nc_var.dimensions, dims
    ))

  order = [
    nc_var.dimensions.index(d) for d in dims
  ]
  return np.transpose(nc_var, order)
dataSingle = dict()
for var in variables4D:
    # target dimensions: lon, lat, level
    d = _transpose_nc_var_by_dims(thermodyn.variables[var], ('time', 'lon', 'lat', 'height'))
    dataSingle[var] = d[forecastIndex, :, :, ::-1] #reverse height order
#%%


pamData = dict()

# time and location
pamData["timestamp"] = time[:]
pamData["lat"] =  lats[:]
pamData["lon"] =  lons[:]

# surface propertiesprin
pamData["groundtemp"] = t_g[:]
pamData["sfc_slf"] = fr_land[:]
pamData["sfc_sif"] = np.zeros(pamData['groundtemp'].shape)[:] #Annahme, dass sea ice 0 ist
pamData["wind10u"] = u10[:]
pamData["wind10v"] = v10[:]
pamData["sfc_type"] = np.around(pamData['sfc_slf'])[:]
pamData["sfc_model"] = np.zeros(pamData['groundtemp'].shape)[:]
pamData["sfc_refl"]  = np.chararray(pamData['groundtemp'].shape)[:]
pamData["sfc_refl"][:] = 'S' # land  'F' # ocean 'L' lambertian, land
#pamData["sfc_type"][(pamData['sfc_type'] == 0) & (pamData['sfc_sif'] > 0)] = 1 Nicht nötig, da Annahme, dass es kein sea ice gibt

# vertical profiles
#pamData["hgt"] = np.array([height,]*len(time))[:,:]
#pamData["hgt"] = hyd.height #z[:,:]

pamData["hgt"] = z[:,:]

pamData["press"] = p[:,:]
pamData["temp"] = t[:,:]
pamData["relhum"] = rh[:,:]
pamData["hydro_q"] = hydro_cmpl[:,:,:]

pamData["obs_height"] = np.zeros([len(time),1,len(flight_levels)])
pamData["obs_height"][:,:,:] = flight_levels
#pamData["obs_height"] = np.full([len(time),1],12500.)[:,:]

#testing dict
print("pamData filled")
#print(for key in pamData.keys(): pamData[key].shape)

# Save
np.save(f'/work/um0203/u301032/PAMTRA_output/PAMTRA-ICON_{DATE}_{output_name}.npy', pamData) 
