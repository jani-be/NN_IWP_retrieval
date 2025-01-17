'''
Comparing pamtra simulation and halo data
'''


#%% Loading packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import fsspec
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
#%% Reading halo data

DATE ="0829"

fs = fsspec.filesystem("ipns")
print(fs.glob("ipns://latest.orcestra-campaign.org/products/HALO/radiometer/*.zarr"))
file_flight_0829="ipns://latest.orcestra-campaign.org/products/HALO/radiometer/HALO-20240829a.zarr"

ds_halo=xr.open_dataset(file_flight_0829,engine="zarr")
ds_halo_iwv=xr.open_dataset("ipns://latest.orcestra-campaign.org/products/HALO/iwv/HALO-20240829a.zarr",
                engine="zarr")

#%% Reading pamtra simulation

file_pamtra = "/work/um0203/u301032/PAMTRA_output/PAMTRA-ICON_0829_test_factor_100.nc"
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
ds_halo.sel(frequency =90).TBs.plot.line(x="time")
plt.show()

#First histograms
ds_halo.sel(frequency =90).TBs.plot.hist()
plt.show()

ds_pamtra.sel(frequency =90,outlevel=2).tb.plot.hist()
plt.show()

# Histograms of halo and pamtra for all frequencies 
for freq in np.asarray(ds_halo.frequency):
  plt.hist([ds_pamtra.sel(frequency =freq,outlevel=1).tb,ds_halo.sel(frequency =freq).TBs],density=True,label=["pamtra","halo"])
  plt.legend()
  plt.title(str(freq))
  plt.show()


# Logarithmic Histograms of halo and pamtra for all frequencies 
for freq in np.asarray(ds_halo.frequency):
  plt.hist([ds_pamtra.sel(frequency =freq,outlevel=1).tb,ds_halo.sel(frequency =freq).TBs],density=True,label=["pamtra","halo"],log=True)
  plt.legend()
  plt.title(str(freq))
  plt.show()


# Histogram of halo and pamtra for a chosen frequency
freq =90
plt.hist([ds_pamtra.sel(frequency =freq,outlevel=1).tb,ds_halo.sel(frequency =freq).TBs],density=True,label=["pamtra TB","halo TB "])
plt.legend()
plt.title(str(freq))
plt.show()

# Scatter Plots TB Pamtra vs TB HAMP
#reducing hamp values
x=ds_halo.sel(frequency =freq).TBs.to_numpy()
data=np.random.choice(x[~np.isnan(x)],3887)

for freq in np.asarray(ds_halo.frequency):
  density_scatter(data,ds_pamtra.sel(frequency =freq,outlevel=1).tb,xlabel="HAMP TB [K]",ylabel="PAMTRA TB [K]",title=freq, bins = [30,30] )

# Scatter IWV from halo and pamtra AND SIMULATIONS
x=ds_halo_iwv.IWV.to_numpy()
data_iwv=np.random.choice(x[~np.isnan(x)],3887)
density_scatter(data_iwv,100*ds_pamtra.sel(outlevel=1).iwv,lim=(0,100),xlabel="HAMP",ylabel="PAMTRA",title="IWV [kg/m^2]", bins = [30,30] )
plt.show()
icon_pwr=ds_icon.isel(time=72).sel(ncells=common_idx).drop_dims("height_2").prw
density_scatter(data_iwv,icon_pwr,lim=(0,100),xlabel="HAMP",ylabel="ICON",title="IWV [kg/m^2]", bins = [30,30] )
plt.show()
density_scatter(icon_pwr,100*ds_pamtra.sel(outlevel=1).iwv,lim=(0,100),xlabel="ICON ",ylabel="PAMTRA",title="IWV [kg/m^2]", bins = [30,30] )
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