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

from glob import glob
from functools import partial

import fsspec
#os.environ['TEXMFHOME'] = '/home/m/m301067/texmf'
#os.environ['PATH'] = '/sw/spack-levante/texlive-live2021-l5o6sw/bin/x86_64-linux:' + os.environ['PATH']
#plt.rcParams.update({'font.size': 13, 'font.family': 'TimesNewRoman', 'text.usetex': True})
#rcParams['axes.linewidth'] = 1.5
#rcParams["axes.formatter.use_mathtext"]
    
# Customize the tick labels to remove the minus sign (its in °W)
def format_longitude(x, pos):
    return f"{abs(int(x))}"




def plotting(filename,ds):
  #uses only left and bottom spines
  # has path for saving
  # has high resolution

  fig, ax = plt.subplots()

  #bins = np.arange(10050,15075,100) # final setting
  #(n50_150, bins50_150, patches) = ax.hist(ds.alt, bins,orientation='horizontal')#,color='lightseagreen')#,density=True

  #heights =[11400.0,12650.0,13000.0,13250.0,13600.0,13850.0,14450.0,15000.0	]
  #plt.hlines(heights,xmin=0,xmax=n50_150.max()+0.1*n50_150.max(),color='black',zorder=0)#, color="mediumslateblue") 
  #plt.vlines(heights,ymin=0,ymax=1, color="red")
  
  plt.ylabel("Height [m]")
  plt.xlabel("Count per bin")


  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  plt.rcParams['figure.dpi'] = 400
  plt.rcParams['savefig.dpi'] = 400
  #plt.savefig(f'/home/u/u301032/orcestra/plots/{filename}.png')
  plt.show()

def plotting_multiple(filename,ds):
  #uses only left and bottom spines
  # has path for saving
  # has high resolution

  # Create a figure with a 4x6 grid of subplots
  fig, axes = plt.subplots(4, 6, figsize=(18, 12))
  axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

  for i, ax in enumerate(axes):
#    freq=22.24
#stacked_pamtra = ds_pamtra.tb.stack(flat_dim = ['grid_x','outlevel'])
#fig, ax = plt.subplots()
#ax.hist([stacked_pamtra.sel(frequency =freq),ds_halo.sel(frequency =freq).TBs_filtered],density=True,label=["pamtra","halo"])



    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
  plt.ylabel("Height [m]")
  plt.xlabel("Count per bin")
  plt.rcParams['figure.dpi'] = 400
  plt.rcParams['savefig.dpi'] = 400
  #plt.savefig(f'/home/u/u301032/orcestra/plots/{filename}.png')
  plt.show()


# %%
def only_tropics(ds):
  ds=ds.sel(time=slice('2024-08-01','2024-10-01'))
  return ds

def reading_halo_data():
  #DATE ="0829"

  fs = fsspec.filesystem("ipns")
  ##print(fs.glob("ipns://latest.orcestra-campaign.org/products/HALO/radiometer/*.zarr"))
  #file_flight_0829="ipns://latest.orcestra-campaign.org/products/HALO/radiometer/HALO-20240829a.zarr"
  #file_altitude = 'ipns://latest.orcestra-campaign.org/products/HALO/position_attitude/HALO-20240829a.zarr'

  #ds_halo=xr.open_dataset(file_flight_0829,engine="zarr")
  ds_halo=xr.open_dataset("ipfs://bafybeicbj76n3hi52pxtcyzu5in7efk36fk7lavauishclybrsbvlrpq3e", engine="zarr")#radiometer
  #ds_halo_altitude = xr.open_dataset(file_altitude,engine ="zarr")
  ds_halo_altitude = xr.open_dataset("ipfs://bafybeias3h5uxtt4ky4d4gn6l6gxjqfkzbde5jlunya6g3umnkvn7xoyoe", engine="zarr") #altitude
  ds_halo_iwv_KW=xr.open_dataset("ipfs://bafybeicahqvp4lovuqpu63euo5kbc22sdq4jp5p6h6wib373x72ki34tiu", engine="zarr")#IWV from KW band
  #ds_halo_iwv_KW=xr.open_dataset("ipns://latest.orcestra-campaign.org/products/HALO/iwv/HALO-20240829a.zarr",
  #                engine="zarr")
  ds_sondes = xr.open_dataset("ipfs://bafybeicb33v6ohezyhgq5rumq4g7ejnfqxzcpuzd4i2fxnlos5d7ebmi3m", engine="zarr")#dropsondes
  ds_radar = xr.open_dataset("ipfs://bafybeigmd3dovwm45ylfqxnn2jphsrdjl2jt3dfytv7grkyhleaq42jthe", engine="zarr") #MIRA Cloud Radar Moments

  return only_tropics(ds_radar),only_tropics(ds_halo),only_tropics(ds_halo_altitude),only_tropics(ds_halo_iwv_KW),ds_sondes

def sel_target_date(target_date,ds):
  target_datetime = pd.to_datetime(target_date)
  if 'time' in ds.coords:
     ds_filtered=ds.sel(time=slice(target_datetime, target_datetime + pd.Timedelta(days=1)))
  elif 'sonde_time' in ds.coords:
    ds_filtered = ds.where(ds['sonde_time'].dt.date == xr.DataArray(pd.to_datetime(target_date).date()), drop=True)
  else:
    print("please check dimension names for time dimension")
  return ds_filtered

def add_noise(clean_signal,sigma=0.75):
    
    mu = 0
    
    noise = np.random.normal(mu, sigma, clean_signal.shape) 
    noisy_signal = clean_signal + noise
    
    return noisy_signal

def get_HAMP_freqs_of(select='all_2side'):
    """
    Function to return frequenices of specified HAMP channel(s).
    """
  
    HAMP_freqs = {'K_band':[22.24,23.04,23.84,25.44,26.24,27.84,31.40],
                  'V_band':[50.30,51.76,52.8,53.75,54.94,56.66,58.00],
                  'W_band':[90.00],
                  'F_band_1side':[118.75+1.4,118.75+2.3,118.75+4.2,118.75+8.5],
                  'F_band_2side':[118.75-8.5,118.75-4.2,118.75-2.3,118.75-1.4,
                                  118.75+1.4,118.75+2.3,118.75+4.2,118.75+8.5],
                  'G_band_1side':[183.31+0.6,183.31+1.5,183.31+2.5,183.31+3.5,183.31+5.0,183.31+7.5,183.31+12.5],
                  'G_band_2side':[183.31-12.5,183.31-7.5,183.31-5.0,183.31-3.5,183.31-2.5,183.31-1.5,183.31-0.6,
                                  183.31+0.6,183.31+1.5,183.31+2.5,183.31+3.5,183.31+5.0,183.31+7.5,183.31+12.5],
                  'NN_freqs':[22.24,23.04,23.84,25.44,26.24,27.84,31.40,
                              50.30,51.76,52.8,53.75,54.94,56.66,58.00,
                              90.00,
                              118.75+1.4,118.75+2.3,118.75+4.2,118.75+8.5,
                              183.31+0.6,183.31+2.5,183.31+3.5,183.31+5.0,183.31+7.5],
                 }
    
    
    if select == 'all_1side':
        freqs = np.concatenate((
            np.array(HAMP_freqs['K_band']),
            np.array(HAMP_freqs['V_band']),
            np.array(HAMP_freqs['W_band']),
            np.array(HAMP_freqs['F_band_1side']),
            np.array(HAMP_freqs['G_band_1side']),))
            
    elif select == 'all_2side':
        freqs = np.concatenate((
            np.array(HAMP_freqs['K_band']),
            np.array(HAMP_freqs['V_band']),
            np.array(HAMP_freqs['W_band']),
            np.array(HAMP_freqs['F_band_2side']),
            np.array(HAMP_freqs['G_band_2side']),))
   
    else:
        freqs = HAMP_freqs[select]
            
    return np.array(freqs)


def create_pamtra_TB_vector(pamtra_ds,outlevels):

    # select "nadir looking" BTs
    pamtra_ds = pamtra_ds.sel(angles=180,grid_y=0)
    pamtra_ds = pamtra_ds.drop(['grid_y','angles'])
    # average over v and h polarisation
    pamtra_ds = pamtra_ds.mean(dim='passive_polarisation')
    # get indices of specified altitudes
    level_inds = [np.where(pamtra_ds.outlevels.values[0,:].squeeze() == level)[0][0] for level in outlevels]
    # select pamtra dataset at specified altitudes
    pamtra_ds = pamtra_ds.sel(outlevel=xr.DataArray(level_inds,dims=['outlevel']))
    
    # select arrays of BTs of K,V,W band
    K_band = pamtra_ds.tb.sel(frequency=get_HAMP_freqs_of('K_band')).values[:,:,:]
    V_band = pamtra_ds.tb.sel(frequency=get_HAMP_freqs_of('V_band')).values[:,:,:]
    W_band = pamtra_ds.tb.sel(frequency=get_HAMP_freqs_of('W_band')).values[:,:].reshape(pamtra_ds.tb.values.shape[0],len(outlevels),1)
    
    # average over doubleside frequencies of F_band
    TB_120_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([118.75-1.4, 118.75+1.4],dims='frequency')),axis=2)
    TB_121_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([118.75-2.3, 118.75+2.3],dims='frequency')),axis=2)
    TB_122_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([118.75-4.2, 118.75+4.2],dims='frequency')),axis=2)
    TB_127_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([118.75-8.5, 118.75+8.5],dims='frequency')),axis=2)
    # create array of BTs of F_band
    F_band = np.empty([K_band.shape[0],len(outlevels),4])
    F_band[:,:,0] = TB_120_mean
    F_band[:,:,1] = TB_121_mean
    F_band[:,:,2] = TB_122_mean
    F_band[:,:,3] = TB_127_mean

    # average over doubleside frequencies of G_band
    TB_183_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-0.6, 183.31+0.6],dims='frequency')),axis=2)
    TB_184_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-1.5, 183.31+1.5],dims='frequency')),axis=2)
    TB_185_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-2.5, 183.31+2.5],dims='frequency')),axis=2)
    TB_186_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-3.5, 183.31+3.5],dims='frequency')),axis=2)
    TB_188_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-5.0, 183.31+5.0],dims='frequency')),axis=2)
    TB_190_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-7.5, 183.31+7.5],dims='frequency')),axis=2)
    TB_195_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-12.5, 183.31+12.5],dims='frequency')),axis=2)
    # create array of BTs of G_band
    G_band = np.empty([K_band.shape[0],len(outlevels),5])
    G_band[:,:,0] = TB_183_mean
    G_band[:,:,1] = TB_185_mean
    G_band[:,:,2] = TB_186_mean
    G_band[:,:,3] = TB_188_mean
    G_band[:,:,4] = TB_190_mean
    #G_band = replace_outliers_with_CHmean(G_band,lower_thrs=230)

    TB_vector = np.concatenate((
        K_band,
        V_band,
        W_band,
        F_band,
        G_band),
        axis=2)
    
    #print("\nCreated ",TB_vector.shape, " TB input vector")
    return TB_vector

def _preprocess(x,cells,frequency="2h"):#, end_time="48h" ):
  # selecting variables of interest
  variables = [
    'prw', #water vapor path
    'qivi', #cloud ice path
    'cllvi', #cloud liquid water path
    'qrvi', #rain path
    'qsvi', #snow path
    'qgvi' #graupel path
  ]
  x=x[variables]
  
  if pd.Timedelta((x.time[-1]-x.time[0]).values,'h')<pd.Timedelta(25,'h'):
    start_time ='12h'
    t_periods = 4 # for old pamtra runs
  else:
    start_time = '24h'  
    t_periods = 5  # for old pamtra runs
  # selecting time
  start=x.time[0].values+pd.Timedelta(start_time)
  stop=x.time[-1].values #last time step
  t_steps =xr.date_range(start,stop,freq=frequency)
  if frequency=="4h":
      t_steps =xr.date_range(x.time[0].values+pd.Timedelta("12h"),periods=t_periods,freq="4h")  # for old pamtra runs
  #print(start)
  return x.sel(time=t_steps,ncells=cells)

def read_example_2d_file(DATE="0829",appendix = "-high3Drate",preprocess=False,cells=[],frequency="2h"):
    path_sim = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/"
    twodim_file = path_sim + f"orcestra_1250m_{DATE+appendix}/" + f"orcestra_1250m_{DATE+appendix}_atm_2d_ml_DOM01_2024{DATE}T000000Z.nc"
    if preprocess:
        ds= _preprocess(xr.open_dataset(twodim_file),cells,frequency)
    else:
        ds =xr.open_dataset(twodim_file)
    return ds

def load_nn_training_data_pamtra(pamtra_file_names,altitude=12500,noise=True):
    
    #print("Loading PAMTRA training data (TBs)...")
    # create list of all pamtra simulations of retrieval database
    # create list of all pamtra simulations of retrieval database 
    
    #pamtra_files = sorted(glob('/work/um0203/u301238/PAMTRA/PAMTRA_NN_training_data/PAMTRA-ICON_2022041*_4000rndm-profiles_all_hamp_freqs_v3.nc'))
    pamtra_files = sorted(glob(pamtra_file_names))


    # open them as one concatenated multifile dataset
    pamtra = xr.open_mfdataset(
        pamtra_files,
        combine='nested',
        concat_dim='grid_x')

    # create a (profile,frequency) TB input vector out of the PAMTRA simulated TBs 
    # by averaging over all doubleside frequencies
    TB_input_vector = create_pamtra_TB_vector(pamtra,outlevels=[altitude])
    TB_input_vector = TB_input_vector[:,0,:]
    if noise == True:
        # Add random noise to the simulated TBs
        for channel in range(TB_input_vector.shape[1]):
            if (channel >= 0) & (channel <= 6): # K-Band
                TB_input_vector[:,channel] = add_noise(TB_input_vector[:,channel],sigma=0.1)
            if (channel >= 7) & (channel <= 13): # V-Band
                TB_input_vector[:,channel] = add_noise(TB_input_vector[:,channel],sigma=0.2)
            if channel == 14: # W-Band
                TB_input_vector[:,channel] = add_noise(TB_input_vector[:,channel],sigma=0.25)
            if (channel >= 15) & (channel <= 18): # F-Band
                TB_input_vector[:,channel] = add_noise(TB_input_vector[:,channel],sigma=0.6)
            if (channel >= 19) & (channel <= 23): # G-Band
                TB_input_vector[:,channel] = add_noise(TB_input_vector[:,channel],sigma=0.6)

    # Load in numpy arrays containing ICON hydrometeor contents of PAMTRA simulations       # TODO change to 2D files, select
    return TB_input_vector 

def load_nn_training_data_icon(dates,appendices,cell_selection,time_selection):
    
 
    '''
    Insert here hydrometer from 2D and selection
    '''
    # Getting list of 2D icon files to use
    path_sim = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/"
    twodim_files=[]
    for DATE, appendix in zip(dates,appendices):
        twodim_files.append(path_sim + f"orcestra_1250m_{DATE+appendix}/orcestra_1250m_{DATE+appendix}_atm_2d_ml_DOM01_2024{DATE}T000000Z.nc")

    #Reading and First Processing of datasets
    partial_func = partial(_preprocess,cells=cell_selection,frequency=time_selection) 
    ds_icon_2d= xr.open_mfdataset(twodim_files,preprocess=partial_func)#, chunks={"ncells": -1})#,chunks="auto", parallel=True)

    #ICON_arrays = np.load(ICON_array_list[0])
    #for i in range(1,len(ICON_array_list)):
    #    ICON_arrays = np.concatenate((ICON_arrays,np.load(ICON_array_list[i])),axis=0)
    t_steps=ds_icon_2d.time.values
    IWV= np.concatenate(([(ds_icon_2d.prw.sel(time=t))for t in t_steps ]),axis=0) #water vapor path

    # cloud ice
    #IWP= np.concatenate(([(ds_icon_2d.qivi.sel(time=t))for t in t_steps ]),axis=0)
    # liquid water
    #LWP= np.concatenate(([(ds_icon_2d.cllvi.sel(time=t))for t in t_steps ]),axis=0)

    qivi= np.concatenate(([(ds_icon_2d.qivi.sel(time=t))for t in t_steps ]),axis=0) #cloud ice path
    qgvi= np.concatenate(([(ds_icon_2d.qgvi.sel(time=t))for t in t_steps ]),axis=0) #graupel path
    qsvi= np.concatenate(([(ds_icon_2d.qsvi.sel(time=t))for t in t_steps ]),axis=0) #snow path
    IWP= np.sum([qivi,qgvi,qsvi], axis=1)

    cllvi= np.concatenate(([(ds_icon_2d.cllvi.sel(time=t))for t in t_steps ]),axis=0)  #cloud liquid water path
    qrvi= np.concatenate(([(ds_icon_2d.qrvi.sel(time=t))for t in t_steps ]),axis=0) #rain path
    LWP = np.sum([cllvi,qrvi], axis=1)

    return IWV,IWP,LWP,qivi,qgvi,qsvi,cllvi,qrvi,t_steps #, frozen_water, liquid_water, IWV











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

def pamtra_TBs_all_heights_combined(name_pamtra_run,flight_levels =[11400,12650,13000,13250,13600,13850,14450,15000] ):
  altitude=flight_levels[0]
  TBs=np.load('/work/um0203/u301032/master_thesis/ML_input/' + name_pamtra_run + '_TBs_altitude_' + str(altitude) + 'm.npy')
  for altitude in flight_levels[1:]:
      TB=np.load('/work/um0203/u301032/master_thesis/ML_input/' + name_pamtra_run + '_TBs_altitude_' + str(altitude) + 'm.npy')
      TBs=np.concatenate([TBs,TB])
  return TBs

def halo_BT_read_in():
  ds_radar,ds_halo,ds_halo_altitude,ds_halo_iwv_KW,ds_sondes=reading_halo_data()
  
  #ds_halo no sea please
  da_masked = ds_halo.TBs.where(ds_halo.mask_sea_land, drop=True)

  # use amplifier fault mask and land sea mask
  da_masked = da_masked.where(ds_halo.mask_amplifier_fault, drop=True)

    # filter 90 GHz channel for unrealitstic temperatures
  mask_90=(ds_halo.sel(frequency=90.).TBs<320).drop_vars('frequency') # 38  caases over 320
  da_masked = da_masked.where(mask_90, drop=True)
  #cut altitude:
  da_masked = da_masked.where(ds_halo.plane_altitude>=10500,drop=True)

  # add filtered data array to variable
  ds_halo['TBs']=  da_masked
  excluded_frequencies = ds_halo.frequency.where(ds_halo.frequency!=184.81, drop=True)
  # Step 2: Filter the dataset to exclude the specified frequency
  ds_halo = ds_halo.sel(frequency=excluded_frequencies)
  return ds_halo

def pamtra_BT_read_in(pamtra_file_names):
  pamtra_files = sorted(glob(pamtra_file_names))

  # open them as one concatenated multifile dataset
  ds_pamtra = xr.open_mfdataset(
      pamtra_files,
      combine='nested',
      concat_dim='grid_x')

  # select "nadir looking" BTs
  ds_pamtra = ds_pamtra.sel(angles=180,grid_y=0)
  ds_pamtra = ds_pamtra.drop_vars(['grid_y','angles'])
  # average over v and h polarisation
  ds_pamtra = ds_pamtra.mean(dim='passive_polarisation')

  return ds_pamtra


def pamtra_halo_BT_plot(ds_halo, ds_pamtra,freq=90.):
  #
  
  # For the histograms we want data from all outlevels
  stacked_pamtra = ds_pamtra.tb.stack(flat_dim = ['grid_x','outlevel'])
  plt.xlabel('BTs [K]')
  plt.hist([stacked_pamtra.sel(frequency =freq),ds_halo.sel(frequency =freq).TBs],bins=np.arange(np.min(ds_halo.TBs),np.max(ds_halo.TBs)),density=True,label=["pamtra","halo"])
  
  plt.legend()
  plt.title(str(freq))
  plt.show()

def plotting_multiple_with_noise(filename=None,density=True,log=False):

  # Creates plot of all frequenciesHstograms of BTs PAMTRA vs HAMP
  # uses only left and bottom spines
  # has path for saving
  # has high resolution

  # read in PAMTRA and HAMP files

  ds_halo = halo_BT_read_in()
  name_pamtra_run = "cells_025x025_2h"#"all_area_1000th_cell" # 
  TBs = pamtra_TBs_all_heights_combined(name_pamtra_run)


  plt.rcParams['font.size'] = '16'
  # Create a figure with a 6x4 grid of subplots
  fig, axes = plt.subplots(6, 4, sharex=True,sharey=True, figsize=(16, 24))
  #fig, axes = plt.subplots(6, 4,figsize=(16, 24))
  axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
  maxval = max(np.nanmax(ds_halo.TBs),np.nanmax(TBs))
  minval = min(np.nanmin(ds_halo.TBs),np.nanmin(TBs))
  bins=np.arange(minval,maxval,20)#(np.min(ds_halo.TBs),np.max(ds_halo.TBs),20)
  for i, ax in enumerate(axes):
    freq=np.array(ds_halo.frequency)[i]
    ax.hist([TBs[:,i],ds_halo.sel(frequency =freq).TBs],bins=bins,density=density,log=log,label=["pamtra","halo"])
    # calculate bias and plot bias
    bias=np.mean(TBs[:,i])-np.mean(ds_halo.sel(frequency =freq).TBs)
    ax.text(0.05, 0.8, str(np.round(bias.values,2))+' K', transform=ax.transAxes,
      fontsize=20,  va='top')
    ax.text(0.05, 1.0, str(freq)+ ' GHz', transform=ax.transAxes,
      fontsize=20,  va='top')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
  fig.supxlabel('Brightness Temperatures \\K')
  plt.tight_layout()
  plt.rcParams['figure.dpi'] = 400
  plt.rcParams['savefig.dpi'] = 400
  if filename != None:
      plt.savefig(f'/home/u/u301032/orcestra/plots/BT_pamtra_hamp_noise_height_{filename}.png')
  plt.show()
  
def plotting_fav4_noise(filename=None,density=True,log=False,indices=[0,14,19,23]):

  # Creates plot of all frequenciesHstograms of BTs PAMTRA vs HAMP
  # uses only left and bottom spines
  # has path for saving
  # has high resolution

  # read in PAMTRA and HAMP files

  ds_halo = halo_BT_read_in()
  name_pamtra_run = "cells_025x025_2h"#"all_area_1000th_cell" # 
  TBs = pamtra_TBs_all_heights_combined(name_pamtra_run)


  plt.rcParams['font.size'] = '16'
  # Create a figure with a 6x4 grid of subplots
  ##fig = plt.figure(figsize=(16, 8),constrained_layout=True)
  fig = plt.figure(figsize=(16, 8),constrained_layout=True)
  subfigs = fig.subfigures(1, 4)
  ##fig, axes = plt.subplots(6, 4, figsize=(16, 24))
  #fig, axes = plt.subplots(6, 4,figsize=(16, 24))
  ##axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
  maxval = max(np.nanmax(ds_halo.TBs),np.nanmax(TBs))
  minval = min(np.nanmin(ds_halo.TBs),np.nanmin(TBs))
  bins=np.arange(minval,maxval,20)#(np.min(ds_halo.TBs),np.max(ds_halo.TBs),20)
  for i, subfig, title in zip(indices,subfigs.flat, ['a)','b)','c)','d)']):     
    freq=np.array(ds_halo.frequency)[i]
    #subfig.suptitle(' ',ha='left')
    subfig.text(0.0, 1.0, title, #transform=ax.transAxes,
                               fontsize='large',  va='top')
    axs = subfig.subplot_mosaic([['lin'],['log']],
                                  height_ratios=(1, 1), sharex=True)

     
    axs['lin'].hist([TBs[:,i],ds_halo.sel(frequency =freq).TBs],bins=bins,density=density,log=False,label=["pamtra","halo"])
    # calculate bias and plot bias
    bias=np.mean(TBs[:,i])-np.mean(ds_halo.sel(frequency =freq).TBs)
    if bias >= 0:
      label= '+' + str(np.round(bias.values,2))+' K'
    else:
      label= str(np.round(bias.values,2))+' K'
    axs['lin'].set_title(str(freq)+ ' GHz', loc='left', fontsize='large',weight='semibold')
    #axs['lin'].text(0.05, 0.1, str(np.round(bias.values,2))+' K',# transform=axs['lin'].transAxes,
    #  fontsize=20,  va='top')
    #axs['lin'].text(0.05, 0.5, str(freq)+ ' GHz', #transform=axs['lin'].transAxes,
    #  fontsize=20,  va='top')
    axs['lin'].annotate(
        label,
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top',color='slategray')

    axs['log'].hist([TBs[:,i],ds_halo.sel(frequency =freq).TBs],bins=bins,density=density,log=True,label=["pamtra","halo"])
    # calculate bias and plot bias
    axs['lin'].set_ylim(top=0.05)
    axs['log'].set_ylim(top=0.1,bottom=2*10**(-8))
    axs['log'].spines['right'].set_visible(False)
    axs['log'].spines['top'].set_visible(False)
    axs['lin'].spines['right'].set_visible(False)
    #axs['lin'].spines['top'].set_visible(False)
    subfig.supxlabel('BT \\ K')
  #plt.tight_layout()
  plt.rcParams['figure.dpi'] = 400
  plt.rcParams['savefig.dpi'] = 400
  if filename != None:
      plt.savefig(f'/home/u/u301032/orcestra/plots/BT_pamtra_hamp_noise_height_fav4_{filename}.png')
  plt.show()

def plotting_multiple_no_noise(filename=None,density=True,log=False):

  # Creates plot of all frequenciesHstograms of BTs PAMTRA vs HAMP
  # uses only left and bottom spines
  # has path for saving
  # has high resolution

  # read in PAMTRA and HAMP files
  pamtra_files='/work/um0203/u301032/PAMTRA_output/PAMTRA-ICON_0829*_025x025_2h_v1.nc'

  ds_halo = halo_BT_read_in(pamtra_files)
  ds_pamtra = pamtra_BT_read_in(pamtra_files)


  plt.rcParams['font.size'] = '16'
  # Create a figure with a 6x4 grid of subplots
  fig, axes = plt.subplots(6, 4, sharex=True,sharey=True, figsize=(16, 24))
  #fig, axes = plt.subplots(6, 4,figsize=(16, 24))
  axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
  # Stacking pamtra array in order to include all pamtra outputlevels
  stacked_pamtra = ds_pamtra.tb.stack(flat_dim = ['grid_x','outlevel'])
  bins=np.arange(np.min(ds_halo.TBs),np.max(ds_halo.TBs),20)
  for i, ax in enumerate(axes):
    freq=np.array(ds_halo.frequency)[i]
    ax.hist([stacked_pamtra.sel(frequency =freq),ds_halo.sel(frequency =freq).TBs],bins=bins,density=density,log=log,label=["pamtra","halo"])
    
    ax.text(0.05, 1.0, str(freq), transform=ax.transAxes,
      fontsize=20,  va='top')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
  fig.supxlabel('Brightness Temperatures \\K')
  plt.tight_layout()
  plt.rcParams['figure.dpi'] = 400
  plt.rcParams['savefig.dpi'] = 400
  if filename != None:
      plt.savefig(f'/home/u/u301032/orcestra/plots/{filename}.png')
  plt.show()


# %%
