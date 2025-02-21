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
from functools import partial
#%% Reading halo data

DATE ="0829"

fs = fsspec.filesystem("ipns")
#print(fs.glob("ipns://latest.orcestra-campaign.org/products/HALO/radiometer/*.zarr"))
file_flight_0829="ipns://latest.orcestra-campaign.org/products/HALO/radiometer/HALO-20240829a.zarr"
file_altitude = 'ipns://latest.orcestra-campaign.org/products/HALO/position_attitude/HALO-20240829a.zarr'

ds_halo=xr.open_dataset(file_flight_0829,engine="zarr")

ds_halo_altitude = xr.open_dataset(file_altitude,engine ="zarr")
ds_halo_iwv=xr.open_dataset("ipns://latest.orcestra-campaign.org/products/HALO/iwv/HALO-20240829a.zarr",
                engine="zarr")


#%% Data Comparison condensate loads all campaign days with reruns (2D field)
#Read in all 2D fields.
#How to read multiple files?
DATE="0829"
appendix = ""

path_sim = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/"
path = path_sim + f"orcestra_1250m_{DATE+appendix}/"
twodim_file = path + f"orcestra_1250m_{DATE+appendix}_atm_2d_ml_DOM01_2024{DATE}T000000Z.nc"
twodim_files  = path_sim + "orcestra_1250m_*[0-9]/" +"orcestra_1250m_*_atm_2d_ml_DOM01_2024*T000000Z.nc"



def _preprocess(x, start_time="24h"):#, end_time="48h" ):
  start=x.time[0].values+pd.Timedelta(start_time)
  #stop=x.time[0].values +pd.Timedelta(end_time) #last time step
  stop=x.time[-2].values #last time step
  t_steps =xr.date_range(start,stop,freq="2h")
  print(start)
  return x.sel(time=t_steps)
#How to determine time step 
partial_func = partial(_preprocess,start_time="24h") #, end_time="48h")
#ds= xr.open_dataset(twodim_file)
ds_alldays= xr.open_mfdataset(twodim_files,preprocess=partial_func)
# take out time steps erlier than 12:00h
#plot histogram of condensate loads for all days together
# Condensate loads of interest: IWV, IWP, PP
# -> 3 plots
# plot histogram of the 3 rerun days combined
# -> 3 plots
# merge upper 6 plots into 3 plots
# check, if upper and lower ends of rerun days extent to same variability as all days combined
# if not, think about how to choose which days to take extra in account
# if yes, check if subsampled area (100th icon) has same variability

#print(fs.glob("ipns://latest.orcestra-campaign.org/products/HALO/iwv/*.zarr"))
#%%
#ds4 =ds.chunk(dict(time=-1))

#%% Halo Ice peak analysis
# which flights are particulary interesting? From all and the golden days
# how well are the reruns?

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




#%% Ermitteln der Flughöhen
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

#%% Reading pamtra simulation

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