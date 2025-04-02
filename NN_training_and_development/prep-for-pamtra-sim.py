import numpy as np
import xarray as xr
from netCDF4 import Dataset
import random
import sys
sys.path.append('/home/u/u301032/orcestra/NN_IWP_retrieval/')
import src_comparison_halo_pamtra as chp
from functools import partial
import pandas as pd

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
try:
    sys.argv
except:
    print("no DATE was given. Using now DATE",DATE)
else:
    DATE = (sys.argv[1])
    print("pamtra simulation for DATE",DATE)    
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



height = xr.open_dataset(height_file)
frac_land= xr.open_dataset(frac_land_file, chunks={"cell": -1})
frac_land=frac_land.drop_dims("nv")
# %%%

# Limiting simulation data from 24h - 48 h
def _preprocess(x, start_time="24h"):#, end_time="48h" ):
  if DATE == "0829":
      start_time = "12h"
  start=x.time[0].values+pd.Timedelta(start_time)
  #stop=x.time[0].values +pd.Timedelta(end_time) #last time step
  stop=x.time[-2].values #last time step
  t_steps =xr.date_range(start,stop,freq="2h")
  #print(start)
  return x.sel(time=t_steps)

partial_func = partial(_preprocess,start_time="24h") #, end_time="48h")
twodim= _preprocess(xr.open_dataset(twodim_file, chunks={"ncells": -1}))
thermodyn =_preprocess(xr.open_dataset(twodim_file, chunks={"ncells": -1}))#xr.open_dataset(thermodyn_file)

if DATE == "0829":
    hyd1=  _preprocess(xr.open_dataset(hydrometeor1_file, chunks={"ncells": -1}))
    hyd2=  _preprocess(xr.open_dataset(hydrometeor2_file, chunks={"ncells": -1}))
    hyd=    xr.merge([hyd1,hyd2])
else:
    hyd = _preprocess(xr.open_dataset(hydrometeor_file, chunks={"ncells": -1}))

frac_land=chp.remap(frac_land,input_core_dim="cell")
frac_land=chp.coarse_for_pamtra(frac_land,res=0.25)
#grid,frac_land = chp.read_grid_and_cell_data()
def prep_ds(ds):
    ds=chp.remap(ds)
    ds=chp.cut_to_area(ds)
    ds=chp.coarse_for_pamtra(ds,res=0.25)
    ds.where((frac_land.sea==1).compute(),drop=True) 
    
    return ds

twodim=prep_ds(twodim)
thermodyn=prep_ds(thermodyn)
hyd = prep_ds(hyd.drop_vars('height_bnds'))


height # nur Höhe entnehmen. Kein remappen nötig
frac_land # still needs mask and coarsing


# only ocean data

########
#%%
#choose subset tp work with
ds_small=chp.coarse_for_pamtra(ds_sea,res=0.25)


#processing of all sorts



# save as nparray