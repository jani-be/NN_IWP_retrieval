'''
Comparing pamtra simulation and halo data
plus choosing of flight levels
'''


#%% Loading packages
import numpy as np

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import fsspec
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import glob
from functools import partial
import sys
#sys.path.append('/home/u/u301238/master_thesis/')
sys.path.append('/home/u/u301032/orcestra/NN_IWP_retrieval/')
#import src
import src_comparison_halo_pamtra as chp

#%% Data Comparison condensate loads all campaign days with reruns (2D field)
#Read in all 2D fields.
'''
Comparing pamtra simulation and halo data
plus choosing of flight levels
'''

#ds_remap.coarsen(lat=1,lon=1,boundary='pad').construct({"lat":("lat_c",lat), "lon":("lon_c",lon)})





#%% Data Comparison condensate loads all campaign days with reruns (2D field)
#Read in all 2D fields.
DATE="0829"
appendix = ""

path_sim = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/"
path = path_sim + f"orcestra_1250m_{DATE+appendix}/"
twodim_file = path + f"orcestra_1250m_{DATE+appendix}_atm_2d_ml_DOM01_2024{DATE}T000000Z.nc"
twodim_files  = path_sim + "orcestra_1250m_*[0-9]/" +"orcestra_1250m_*_atm_2d_ml_DOM01_2024*T000000Z.nc"

# Opening Multifile Dataset
# Limiting simulation data from 24h - 48 h
def _preprocess(x, start_time="24h"):#, end_time="48h" ):
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

  # selecting time
  start=x.time[0].values+pd.Timedelta(start_time)
  #stop=x.time[0].values +pd.Timedelta(end_time) #last time step
  stop=x.time[-2].values #last time step
  t_steps =xr.date_range(start,stop,freq="2h")
  #print(start)
  return x.sel(time=t_steps)


#ds_alldays=xr.concat([xr.open_dataset(i,preprocess=partial_func,engine ="zarr")for i in twodim_files],dim="time")

#twodim_files_list =glob.glob(twodim_files)
#ds_alldays=xr.concat([_preprocess(xr.open_dataset(i))for i in twodim_files_list],dim="time")


def read_grid_and_cell_data():
  meshdir = "/work/mh0492/m301067/orcestra/auxiliary-files/grids/"
  meshname = "ORCESTRA_1250m_DOM01"
  DATE="0829"
  appendix = ""
  path_sim = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/"
  path = path_sim + f"orcestra_1250m_{DATE+appendix}/"
  frac_land_file= path + "bc_land_frac.nc"
  grid = xr.open_dataset(meshdir+meshname+".nc",chunks="auto")
  frac_land= xr.open_dataset(frac_land_file, chunks={"cell": -1})
  frac_land=frac_land.drop_dims("nv")
  return grid,frac_land





division_factor =1000 #1000 leads to ~ 6000 n_spatial

#Reading and First Processing of datasets
partial_func = partial(_preprocess,start_time="24h") #, end_time="48h")
#ds= _preprocess(xr.open_dataset(twodim_file, chunks={"ncells": -1}))
ds_alldays= xr.open_mfdataset(twodim_files,preprocess=partial_func, chunks={"ncells": -1})#,chunks="auto", parallel=True)
grid,frac_land = read_grid_and_cell_data()


if "cell" in set(frac_land.dims): 
    frac_land=chp.remap(frac_land,input_core_dim="cell")
frac_land=chp.cut_to_area(frac_land)

ds_remap=chp.remap(ds_alldays)
ds_remap=chp.cut_to_area(ds_remap)

# only ocean data
ds_sea=ds_remap.where((frac_land.sea==1).compute(),drop=True) 
########
#%%
#choose subset tp work with
ds_small=chp.coarse_for_pamtra(ds_sea,res=0.25)
#ds_small.to_netcdf("/work/um0203/u301032/icon_coarse_nosea_2h_025_025.nc")
#ds_small["prw"].isel(time=0).plot()
#ds_small.isel(time=0).plot.scatter(x="lat",y="lon",marker='.')
#plt.show()


#Max:
#prw 100 #wahrscheinlich zu hoch. sonst 80?
# qivi: 3.619194
#cllvi 21.685883
#qrvi 88.09291
#qgvi 47.29776
#qsvi 17.277763
#bins = np.linspace(0,3.62,num=100)
#(n2, bins2, patches) = plt.hist(ds_alldays.qivi, bins)

variables = [
    'prw', #water vapor path
    'qivi', #cloud ice path
    'cllvi', #cloud liquid water path
    'qrvi', #rain path
    'qsvi', #snow path
    'qgvi' #graupel path
  ]
si_units = []
#hist=[ds_alldays[variable].plot.hist(bins=100) for variable in variables]
ds_bins_alldays = xr.Dataset()
#xarray Grundgerüst erstellen  
bins_all= pd.read_csv("hyd_climatology_n.csv")
for variable in variabl
es:
    bin_values = bins_all["bins_"+variable] # unterschiedlich für jede Größe
    time_values=pd.date_range(start='2024-08-10', end='2024-09-30')
    array =  np.zeros((len(time_values),len(bin_values))) #n2 # shape: time,bins
    da=xr.DataArray(array, dims=("time","bins_"+variable), coords={"bins_"+variable: bin_values, "time": time_values})
    ds_bins_alldays[variable]=da


for v in variables:
    #ds[v]=ds_alldays[v].groupby_bins(bins=bins_all["bins_"+v])#.resample(time='1D').map(digi,var=v)
    #ds[v]=ds_alldays[v].resample(time='1D').map(xr.plot.hist(ds,bins=bins_all["bins_"+v]))
    #a=ds_alldays[v].resample(time='1D').map(hist,var=v)
    #ds[v]=ds_alldays[v].resample(time='1D').map(hist,var=v)
    
    for label, group in ds_small[v].resample(time='1D'):
        h=group.plot.hist(bins=bins_all["bins_"+v])
        ds_bins_alldays[v].loc[label][:-1]=h[0]
ds_bins_alldays.to_netcdf("hyd_climatology_hist_025x025.nc")
#%%
def old_hist():
  ds_alldays[variables].sel(time=[slice("2024-08-24","2024-08-25"),"2024-08-29","2024-09-27"])# testen ob funktioniert
  #Wenn ja, dann:
  hist_reruns=[]
  for day in ["2024-08-24","2024-08-29","2024-09-27"]:

    a=ds_alldays[variables].sel(time=slice(day,pd.Timestamp(day)+pd.Timedelta(23,"h")))
    print(day)
    hist_rerun=[a[variable].plot.hist(bins=100) for variable in variables]
    hist_reruns.append(hist_rerun)
  hist_reruns_np=np.array(hist_reruns,dtype=object)
  #df=pd.DataFrame(data={'n1':hist_reruns[],'bins':,'time':["2024-08-24","2024-08-29","2024-09-27"]})
  #
  #foo = xr.DataArray(np.array(hist_reruns,dtype=object), coords=times, dims=["time","variables","unit"])
  #xr.Dataset(
  #  {
  #    "n": (["time","variable"],np.array(hist_reruns,dtype=object)[:,:,0]),
  #    "bins": (["time","variable"],np.array(hist_reruns,dtype=object)[:,:,1])
  #  },
  #  coords={
  #        "variable": variables,
  #        "time": times,
  #    },
  #)
  times=["2024-08-24","2024-08-29","2024-09-27"]
  for i in range(6):
    #n1=hist_reruns[3*2*i]
    #bins1=hist_reruns[2*i+1]
    
    n2= hist[2*i]
    bins2=hist[2*i+1]
    plt.stairs(n2[1:], bins2[1:],fill=True)
    plt.stairs(hist_reruns_np[0,i,0][1:],hist_reruns_np[0,i,1][1:])
    for day in range(3):
      plt.stairs(hist_reruns_np[day,i,0][1:],hist_reruns_np[day,i,1][1:],label=times[day])#, baseline=hist_reruns_np[day,i,0])
      
    plt.legend()
    plt.yscale("log")
    plt.title(variables[i])
    plt.show()
    print(variables[i],2*i)
    

    #
  hist_reruns=[ds_alldays[variable].sel(time=["2024-08-24","2024-08-29","2024-09-27"]).plot.hist(bins=100) for variable in variables]
  for i in range(6):
    n2= hist[2*i]
    bins2=hist[2*i+1]
    plt.stairs(n2[1:], bins2[1:],fill=True)
    #plt.legend()
    plt.yscale("log")
    plt.title(variables[i])
    plt.show()
    print(variables[i],2*i)
  #plot histogram of condensate loads for all days together
  # Condensate loads of interest: IWV, IWP, PP
  # -> 3 plots
  #bins = np.arange(50,15060,100)
  #(n2, bins2, patches) = plt.hist(ds_alldays.prw,bins=100)
  #plt.stairs(n2, bins2,fill=False)

  # names of variables
  # prw water vapor path
  # qivi cloud ice path
  # cllvi cloud liquid water path
  # qrvi rain path
  # qsvi snow path
  # qgvi graupel path



  # plot histogram of the 3 rerun days combined
  # -> 3 plots
  # merge upper 6 plots into 3 plots
  # check, if upper and lower ends of rerun days extent to same variability as all days combined
  # if not, think about how to choose which days to take extra in account
  # if yes, check if subsampled area (100th icon) has same variability

  #print(fs.glob("ipns://latest.orcestra-campaign.org/products/HALO/iwv/*.zarr"))




#%% Reading halo data
def Reading_halo_data():
  DATE ="0829"

  fs = fsspec.filesystem("ipns")
  #print(fs.glob("ipns://latest.orcestra-campaign.org/products/HALO/radiometer/*.zarr"))
  file_flight_0829="ipns://latest.orcestra-campaign.org/products/HALO/radiometer/HALO-20240829a.zarr"
  file_altitude = 'ipns://latest.orcestra-campaign.org/products/HALO/position_attitude/HALO-20240829a.zarr'

  ds_halo=xr.open_dataset(file_flight_0829,engine="zarr")

  ds_halo_altitude = xr.open_dataset(file_altitude,engine ="zarr")
  ds_halo_iwv=xr.open_dataset("ipns://latest.orcestra-campaign.org/products/HALO/iwv/HALO-20240829a.zarr",
                  engine="zarr")
  return ds_halo,ds_halo_altitude,ds_halo_iwv

#Reading_halo_data()

#%%
#ds4 =ds.chunk(dict(time=-1))

#%% Halo Ice peak analysis
# which flights are particulary interesting? From all and the golden days
# how well are the reruns?
def halo_icepeak_analysis():
  #reading g band data
  ds_hamp_list=[]
  list_a=(fs.glob("ipns://latest.orcestra-campaign.org/products/HALO/radiometer/*"))
  [ds_hamp_list.append(xr.open_dataset("ipns://"+i,
                  engine="zarr"))for i in list_a]
  ds_hamp=xr.concat([xr.open_dataset("ipns://"+i,
                  engine="zarr")for i in list_a],dim="time")
  #log hist of each day

  # Logarithmic Histograms of halo and pamtra for all frequencies 
  for freq in np.asarray([183.91, 184.81, 185.81, 186.81, 188.31, 190.81]):
    plt.hist(ds_hamp.sel(time=slice('2024-08-01','2024-10-01'),frequency =freq).TBs,density=True,log=True,bins=20)
    plt.legend()
    plt.title(str(freq))
    plt.show()

  #22 as first 22 flights of campaign (no Oberpfaffenhofen data)
  freq=183.91
  for flightnr in range(23):
    plt.hist(ds_hamp_list[flightnr].sel(frequency =freq).TBs,density=True,log=True,bins=np.arange(140,320,5))
    #plt.legend()
    #plt.title(str(freq)+" GHz "+ str(ds_hamp_list[flightnr].isel(time=0).time)[-19:-9])
    plt.title(str(ds_hamp_list[flightnr].isel(time=0).time)[-19:-9])
    
    plt.xlim(145,310)
    plt.xlabel("T in K")
    plt.show()

#halo_icepeak_analysis()


#%% Ermitteln der Flughöhen
def flight_altitudes_analysis():
  files_altitude = 'ipns://latest.orcestra-campaign.org/products/HALO/position_attitude/*.zarr'
  #ds=xr.open_mfdataset(files_altitude,engine="zarr")

  ds_altitude=[]
  list_a=(fs.glob("ipns://latest.orcestra-campaign.org/products/HALO/position_attitude/*"))
  [ds_altitude.append(xr.open_dataset("ipns://"+i,
                  engine="zarr"))for i in list_a]
  ds_altitude=xr.concat([xr.open_dataset("ipns://"+i,
                  engine="zarr")for i in list_a],dim="time")
  ds_altitude = ds_altitude.sel(time=slice('2024-08-01','2024-10-01'))

  bins = np.arange(50,15060,100)
  (n2, bins2, patches) = plt.hist(ds_altitude.alt, bins)
  df=pd.DataFrame({"counts":n2,"middle":bins2[:-1]+50})
  df=df[(df["middle"]>=8000)]

  df["rank"]=df["counts"].rank(ascending=False)
  df.sort_values("counts", inplace = True) 

  plt.stairs(n2, bins2,fill=True)
  heights =[15000.0	,12800.0,12500.0,13600.0,12700.0,14000.0,12600.0,11400.0,13000.0,13800.0,13200.0, 14500.0,13300.0,14400.0,13900.0]
  plt.legend()
  plt.title("simulation levels v1")
  plt.xlim(10000,15500)
  plt.vlines(heights,ymin=0,ymax=15000000, color="red")
  plt.show()

  bins = np.arange(9000,15060,30)
  (n2, bins2, patches) = plt.hist(ds_altitude.alt, bins,log=True)#,density=True
  heights =[11900,15000.0	,13600.0,12650.0,11400.0,13000.0,13250.0,14450.0,13900.0]
  plt.vlines(heights,ymin=0,ymax=n2.max(), color="red")
  plt.show()
  plt.stairs(n2, bins2,fill=True)
  heights =[11900,15000.0	,13600.0,12650.0,11400.0,13000.0,13250.0,14450.0,13900.0]
  #plt.legend()
  plt.title("flight levels")
  plt.xlim(11000,15500)
  plt.vlines(heights,ymin=0,ymax=n2.max(), color="red")
  plt.xlabel("height [m]")
  plt.show()

#flight_altitudes_analysis()
#%% Reading pamtra simulation
def pamtra_halo_comparison():
  ds_halo,ds_halo_altitude,ds_halo_iwv =Reading_halo_data()
  file_pamtra = "/work/um0203/u301032/PAMTRA_output/PAMTRA-ICON_0829_test_factor_100_new_rh.nc"

  file_pamtra2 = "/work/um0203/u301032/PAMTRA_output/PAMTRA-ICON_0829_test_factor_100.nc"
  ds_pamtra=xr.open_dataset(file_pamtra, engine="netcdf4")

  # select "nadir looking" BTs
  ds_pamtra = ds_pamtra.sel(angles=180,grid_y=0)
  ds_pamtra = ds_pamtra.drop_vars(['grid_y','angles'])
  # average over v and h polarisation
  ds_pamtra = ds_pamtra.mean(dim='passive_polarisation')

  common_idx=np.load('/work/um0203/u301032/PAMTRA_output/PAMTRA-ICON_{DATE}_test_factor_100_common_idx.npy') 
  ds_pamtra= ds_pamtra.assign_coords({"grid_x":common_idx})

  #%% Reading ICON IWV
  path="/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/orcestra_1250m_0829/"
  file =path+"orcestra_1250m_0829_atm_2d_ml_DOM01_20240829T000000Z.nc"
  ds_icon =  xr.open_dataset(file)
  #%% Functions

  def density_scatter( x , y, ax = None, sort = True, bins = 20,title="title",xlabel="x",ylabel="y",lim=(100,300), **kwargs )   :
      """
      Scatter plot colored by 2d histogram
      """
      
      
      if ax is None :
          fig , ax = plt.subplots()
      ax.axline((0,0),slope=1,color="grey", zorder=1)
      data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
      z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

      #To be sure to plot all data
      z[np.where(np.isnan(z))] = 0.0

      # Sort the points by density, so that the densest points are plotted last
      if sort :
          idx = z.argsort()
          x, y, z = x[idx], y[idx], z[idx]

      ax.scatter( x, y, c=z, **kwargs )

      norm = Normalize(vmin = np.min(z), vmax = np.max(z))
      cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
      ax.set_xlim(lim)
      ax.set_ylim(lim)
      cbar.ax.set_ylabel('Density')
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      ax.set_title(str(title))

      return ax


  #%% Comparison 
  #plotting the values:
  #ds_halo.sel(frequency =90).TBs.plot.line(x="time")
  #plt.show()
  # Calculation of bias
  #bias= np.mean(sim) - np.mean()

  ds_pamtra.sel(outlevel=3).tb.mean("grid_x")
  ds_halo.TBs.mean("time")
  frequencies = ds_halo.frequency
  bias=ds_pamtra.sel(frequency=frequencies,outlevel=3).tb.mean("grid_x")-ds_halo.TBs.mean("time")


  bias.to_pandas()

  #df=pd.DataFrame({"frequency":frequencies,"bias":bias})
  #Calculation of percentiles
  pamtra_percentile=ds_pamtra.sel(frequency=frequencies,outlevel=3).tb.quantile([0.01,0.25,0.50,0.75,0.99],dim="grid_x")
  halo_percentile=ds_halo.TBs.quantile([0.01,0.25,0.50,0.75,0.99],dim="time")

  (halo_percentile.to_pandas())
  (pamtra_percentile.to_pandas())
  print("difference percentile  halo - pamtra")
  (halo_percentile.to_pandas())-(pamtra_percentile.to_pandas())#First histograms
  #ds_halo.sel(frequency =90).TBs.plot.hist()
  #plt.show()

  #ds_pamtra.sel(frequency =90,outlevel=2).tb.plot.hist()
  #plt.show()

  # Histograms of halo and pamtra for all frequencies 
  for freq in np.asarray(ds_halo.frequency):
    plt.hist([ds_pamtra.sel(frequency =freq,outlevel=3).tb,ds_halo.sel(frequency =freq).TBs],density=True,label=["pamtra","halo"])
    plt.legend()
    plt.title(str(freq))
    plt.show()


  # Logarithmic Histograms of halo and pamtra for all frequencies 
  for freq in np.asarray(ds_halo.frequency):
    plt.hist([ds_pamtra.sel(frequency =freq,outlevel=3).tb,ds_halo.sel(frequency =freq).TBs],density=True,label=["pamtra","halo"],log=True)
    plt.legend()
    plt.title(str(freq))
    plt.show()


  # Histogram of halo and pamtra for a chosen frequency
  freq = 58.
  plt.hist([ds_pamtra.sel(frequency =freq,outlevel=3).tb,ds_halo.sel(frequency =freq).TBs],density=True,label=["pamtra TB","halo TB "])
  plt.legend()
  plt.title(str(freq))
  plt.show()

  # Scatter Plots TB Pamtra vs TB HAMP
  #reducing hamp values


  for freq in np.asarray(ds_halo.frequency):
    x=ds_halo.sel(frequency =freq).TBs.to_numpy()
    data=np.random.choice(x[~np.isnan(x)],3887)
    density_scatter(data,ds_pamtra.sel(frequency =freq,outlevel=3).tb,xlabel="HAMP TB [K]",ylabel="PAMTRA TB [K]",title=freq, bins = [30,30] )

  #%%
  # Scatter IWV from halo and pamtra AND SIMULATIONS
  x=ds_halo_iwv.IWV.to_numpy()
  data_iwv=np.random.choice(x[~np.isnan(x)],3887)
  density_scatter(data_iwv,ds_pamtra.sel(outlevel=3).iwv,lim=(0,100),xlabel="HAMP",ylabel="PAMTRA",title="IWV [kg/m^2]", bins = [30,30] )
  plt.show()
  icon_pwr=ds_icon.isel(time=72).sel(ncells=common_idx).drop_dims("height_2").prw
  density_scatter(data_iwv,icon_pwr,lim=(0,100),xlabel="HAMP",ylabel="ICON",title="IWV [kg/m^2]", bins = [30,30] )
  plt.show()
  density_scatter(icon_pwr,ds_pamtra.sel(outlevel=3).iwv,lim=(0,100),xlabel="ICON ",ylabel="PAMTRA",title="IWV [kg/m^2]", bins = [30,30] )
  plt.show()

#pamtra_halo_comparison()
#%% 2D Plot of Pamtra - IWV 2D Plot of simulated IWV 



'''
#%%
#list=listdir("http://127.0.0.1:8080/ipns/latest.orcestra-campaign.org/products/HALO/iwv/",
#                engine="zarr")
ds=xr.open_mfdataset("ipns://latest.orcestra-campaign.org/products/HALO/radiometer/*.zarr",
                engine="zarr")
ds2=xr.open_dataset("ipns://latest.orcestra-campaign.org/products/HALO/iwv/HALO-20240906a.zarr",
                engine="zarr")
#%%
print(ds["Min_TBs"][:,1][:])

#%%
import fsspec
fs = fsspec.filesystem("ipns")
print(fs.glob("ipns://latest.orcestra-campaign.org/products/HALO/iwv/*.zarr"))
#%%
ds4 =ds.chunk(dict(time=-1))
print(ds4)
# perc -> percentile that define the exclusion threshold
# dim -> dimension to which apply the filtering

def replace_outliers(data, dim=0, perc=0.99):

  # calculate percentile
  threshold = data[dim].quantile(perc)

  # find outliers and replace them with max among remaining values
  mask = data[dim].where(abs(data[dim]) <= threshold)
  max_value = mask.max().values
  # .where replace outliers with nan
  mask = mask.fillna(max_value)
  print(mask)
  data[dim] = mask

  return data

ds3 = replace_outliers(ds4, dim="IWV", perc=0.99)
#%%
ds.plot.line("IWV")
ds2.plot.line()
#%%

plt.plot(ds["time"][39000:40000],ds["Min_TBs"][0,39000:40000])
plt.show()

plt.plot(ds["Min_TBs"])
plt.show()


'''