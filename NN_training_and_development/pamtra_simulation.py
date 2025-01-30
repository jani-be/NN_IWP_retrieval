#%%
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import random
import sys
sys.path.append('/home/u/u301032/pamtra/pamtra/python/pyPamtra')

import pyPamtra
from pyPamtra import meteoSI

#%%
# Choose Parameters
#DATE= "0829" 
DATE ="0824" # "0927" #
#NAME UNDER WHICH TO SAFE RUNS
output_name = "for_nn_v1"
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
grid = xr.open_dataset(meshdir+meshname+".nc")

thermodyn =xr.open_dataset(thermodyn_file)
height = xr.open_dataset(height_file)
frac_land= xr.open_dataset(frac_land_file)
#sst = xr.open_dataset(SST_file) #TODO isel timestamp?? zeit falsch angegeben und nur 12 h schritte
twodim = xr.open_dataset(twodim_file)
if DATE == "0829":
    hyd1=  xr.open_dataset(hydrometeor1_file)
    hyd2=  xr.open_dataset(hydrometeor2_file)
    hyd=    xr.merge([hyd1,hyd2])
else:
    hyd = xr.open_dataset(hydrometeor_file)
#print(hyd,thermodyn)
#print(grid)


# %%%
#timestamps for DATE == "0824"
# selection of time stamps: 00,04,08,12,16,20, 24, 28, 32, 36 h
# selection of time = 36:00h

#Für Colocation : von 8 bis 20:00h
day = DATE[2:5]
month = DATE[0:2]
if DATE == "0829":
    t_periods = 4
else:
    t_periods = 10
t_steps =xr.date_range("2024-"+month+"-"+day+" 12:00:00",periods=t_periods,freq="4h")

hyd=hyd.sel(time=t_steps) 
thermodyn=thermodyn.sel(time=t_steps) 
twodim = twodim.sel(time=t_steps)

"""
#Für Colocation : von 8 bis 20:00h
day = str(int(DATE[2:5])+1)
month = DATE[0:2]
t_periods = 7
t_steps =xr.date_range("2024-"+month+"-"+day+" 08:00:00",periods=t_periods,freq="2h")

hyd=hyd.sel(time=t_steps) #sonst 9
thermodyn=thermodyn.sel(time=t_steps) #sonst 9
twodim = twodim.sel(time=t_steps)

#or:

#timestamps for DATE == "0829"
# selection of time stamps: 00,04,08,12,16,20, 24, 28, 32, 36 h
# selection of time = 36:00h

if DATE == "0829":
    thermodyn=thermodyn.isel(time=72) 
    twodim = twodim.isel(time=72)
    hyd = hyd.isel(time=72) # 2024-08-29T12:00:00
"""



#%% Index, spatial

lat = np.rad2deg(grid.clat.to_pandas())

lon=np.rad2deg(grid.clon.to_pandas())

#7.344833, -26.471667
s_lat = 2.5
s_lon = -27.378
n_lon= -24.276
n_lat=18.581

dif_lat = n_lat -s_lat
dif_lon = n_lon -s_lon
m= dif_lat/dif_lon
t = s_lat-m*s_lon

####
lat=lat[(lat <= n_lat )&(lat >= s_lat)]
lon=lon[(lon <= n_lon)&(lon >= s_lon)]
common_idx=lon.index.intersection(lat.index)
#idx_subsample=random.sample(sorted(common_idx.to_numpy()),8000)

idx_list=common_idx.to_list()
#Reducing data size by factor 100
common_idx=idx_list[0::100]
#%% If random sample is wished

'''
#arr = np.zeros(len(common_idx))
#i=0
#while i <= len(common_idx):
#    arr[i] = 1
#    i=i+100

#bool_arr = np.array(arr, dtype='bool') 

#new_idx = common_idx.where(bool_arr).dropna()

#common_idx= new_idx.astype('int64') 
#idx_subsample=sorted(random.sample(sorted(common_idx),8000))
#new_idx = common_idx.where(common_idx.isin(idx_subsample)) #funktioniert nicht

#lat=lat[(lat <= 13)&(lat >= 12.9)]
#lon=lon[(lon <= - 55.9)&(lon >= -56)]
#common_idx = lon.index.intersection(lat.index) 
#SAMPLESIZE= len(common_idx) #80

##
for i_lat in np.arange(s_lat -0.6,n_lat-0.6,1):
    for i_lon in np.arange(s_lon -0.6,n_lon-0.6,1):
        i_lat
        i_lon
        lat_idx=lat[(lat <= i_lat + 1.2)&(lat >= i_lat)]
        lon_idx=lon[(lon <= i_lon +1.2)&(lon >= i_lon)]
        i_common_idx=lon_idx.index.intersection(lat_idx.index)
        len(i_common_idx)
        len(common_idx)
        common_idx=common_idx.union(i_common_idx)


###


#lat=lat[(lat <= 17.5)&(lat >= 2.5)]
#lon=lon[(lon <= n_lon)&(lon >= s_lon)]
#common_idx = lon.index.intersection(lat.index) 
# random.sample(common_idx,4000)
'''
#%%
SAMPLESIZE= len(common_idx) #n_spatial

print(SAMPLESIZE)


#for only onbe timestep
dt_time = hyd.time.values
unix_time = dt_time.astype('datetime64[s]').astype('int')

#calculation of unix time for serveral time steps

#dt_time = hyd.time.values
#unix_time = np.array([
#    dt_time[i].astype('datetime64[s]').astype('int') 
#    for i in range(len(dt_time))])

#Auswahl von Profilen um Faktor 10 kleiner #TODO überlegen wieviele Slices tatsächlich wählen

n_spatial = SAMPLESIZE
time = np.zeros([n_spatial*t_periods])
for i in range(t_periods):
    time[i*n_spatial:(i+1)*n_spatial] = unix_time[i] #unix_time

#%%
hyd=hyd.drop_dims("bnds") 
thermodyn=thermodyn.drop_dims("bnds")
#sst = sst.drop_dims("nv")
frac_land = frac_land.drop_dims("nv")


#hyd=hyd.isel(ncells=common_idx)
hyd=hyd.sel(ncells=common_idx)
thermodyn=thermodyn.sel(ncells=common_idx)

# oder thermodyn=thermodyn.sel(ncells=common_idx)

grid=grid.sel(cell=common_idx)
height=height.sel(ncells=common_idx) 
#sst=sst.isel(cell=common_idx)  # 
twodim = twodim.sel(ncells=common_idx)
frac_land = frac_land.sel(cell = common_idx)

height = height.sel(height_2 = range(35,91))

# seamask beachtne

# give me cells of lon lat values of ....
#ICON_lons = np.array(ICON.clon.values[:])
#lons[t,:] = np.rad2deg(ICON_lons[rndm_ocean_profiles])
#lat = np.deg2rad(12)
#lon = np.deg2rad(-56)
#>>> grid.clon.values[6000]
#-0.9612835272738588

#grid_small = grid.sel(clat=slice(lat,lat+1),clon=slice(lon,lon+1))

#clat_deg = np.rad2deg(grid.clat)
#starting point for just now 12.492500, -56.095667
#slicen und selecten von kleiner Untermenge
#TODO all_ocean_profiles = np.where(frac_land.sea==1)[0].tolist()

# choose a random subsample of these cell indices of SAMPLESIZE
#rndm_ocean_profiles = random.sample(all_ocean_profiles,SAMPLESIZE)

#TODO select / iselect   
#hyd=hyd.sel(ncells=rndm_ocean_profiles)
#thermodyn=thermodyn.sel(ncells=rndm_ocean_profiles)
#grid=grid.sel(ncells=rndm_ocean_profiles)




def flatten_1D_array_manually_dimtime(ds):
    return np.concatenate(([(ds.sel(time=t))for t in t_steps ]),axis=0)
    

def flatten_1D_array_manually(ds):
    arr = ds.values
    return np.concatenate((t_periods*[arr]))
    

def flatten_2D_array_manually_old(ds):
    A = ds.values
    arr=np.fliplr(np.transpose(A)) #das muss ich umdrehem #
    
    #arr = np.zeros([80,56])
    #arr[0:80,:] = ds.isel(time=0).values[:,:]
    #arr[100:200,:] = ds.isel(time=1).values[:,:]
    #arr[200:300,:] = ds.isel(time=2).values[:,:]
    #arr[300:400,:] = ds.isel(time=3).values[:,:]
    #arr[400:500,:] = ds.isel(time=4).values[:,:]
    #arr[500:600,:] = ds.isel(time=5).values[:,:]
    return arr

def flatten_2D_array_manually_old2(ds, dim="height"):
    #arr=np.fliplr(np.transpose(ds.values[:,:])) #das muss ich umdrehem
    #ds =hyd.sel(ncells=common_idx).qv 
    arr = np.zeros([SAMPLESIZE,56])
    h=np.arange(35.,91.,1)
    if dim=="height":
        for i in range(len(h)):
            A=ds.sel(height=h[i]).values  #or ds.isel(height=i).values
            arr[:,len(h)-i-1] = A#np.transpose(A)
            #arr[:,i] = np.transpose(ds.sel(height=h[i]).to_numpy()) #other option
    else:
        for i in range(len(h)):
            A=ds.sel(height_2=h[i]).values  #or ds.isel(height=i).values
            arr[:,len(h)-i-1] = A#np.transpose(A)
            #arr[:,i] = np.transpose(ds.sel(height=h[i]).to_numpy()) #other option 
    return arr

def flatten_2D_array_manually(ds, dim="height"):
    #arr=np.fliplr(np.transpose(ds.values[:,:])) #das muss ich umdrehem
    #ds =hyd.sel(ncells=common_idx).qv 
    arr = np.zeros([SAMPLESIZE*t_periods,56])
    h=np.arange(35.,91.,1)
    for t in range(t_periods):
        if dim=="height":
            for i in range(len(h)):
                A=ds.sel(height=h[i],time=t_steps[t]).values  #or ds.isel(height=i).values
                arr[SAMPLESIZE*t:SAMPLESIZE*(t+1),len(h)-i-1] = A#np.transpose(A)
                #arr[:,i] = np.transpose(ds.sel(height=h[i]).to_numpy()) #other option
        else:
            for i in range(len(h)):
                A=ds.sel(height_2=h[i]).values  #or ds.isel(height=i).values
                arr[SAMPLESIZE*t:SAMPLESIZE*(t+1),len(h)-i-1] = A#np.transpose(A)
                #arr[:,i] = np.transpose(ds.sel(height=h[i]).to_numpy()) #other option 
    return arr

def flipheight(ds,dim="height"):
    arr = np.zeros([SAMPLESIZE,56])
    h=np.arange(35.,91.,1)
    if dim=="height":
        for i in range(len(h)):
            A=ds.sel(height=h[i]).values  #or ds.isel(height=i).values
            arr[:,len(h)-i-1] = A#np.transpose(A)
    return arr

def flatten_2D_array_manually_test(ds, dim="height"):      
    return np.concatenate(([flipheight(ds.sel(time=t),dim)for t in t_steps ]),axis=0)


def flatten_2D_array_manually_height(ds):
    #arr=np.fliplr(np.transpose(ds.values[:,:])) #das muss ich umdrehem
    #ds =hyd.sel(ncells=common_idx).qv 
    arr =  np.zeros([SAMPLESIZE,56])
    h=np.arange(35.,91.,1)
    c=0
    for i in range(len(h)):
        A=ds.sel(height_2=h[i]).values  #or ds.isel(height=i).values
        arr[:,len(h)-i-1] = A
    arr=np.concatenate((t_periods*[arr]))
    return arr

cic = flatten_2D_array_manually(hyd.qi)
#ds =hyd.sel(ncells=common_idx).qv 
#test1=flatten_2D_array_manually(ds)
#test2=flatten_2D_array_manually_old(ds)
#%%
t_g = flatten_1D_array_manually_dimtime(twodim.ts)
#sst = flatten_1D_array_manually(sst.SST) 
u10 = flatten_1D_array_manually_dimtime(twodim.uas)[0]
v10 = flatten_1D_array_manually_dimtime(twodim.vas)[0]

#fr_seaice = flatten_1D_array_manually(ICON_input.fract_glace)

fr_land = flatten_1D_array_manually(frac_land.land)

print("1D arrays are flattened.")



#%%

z = flatten_2D_array_manually_height(height.zg) #geometric_height_at_full_level_center
print("zg has been flattened.")

p = flatten_2D_array_manually(thermodyn.pfull)
t = flatten_2D_array_manually(thermodyn.ta)
#rh = flatten_2D_array_manually(thermodyn.rh) 
cic = flatten_2D_array_manually(hyd.qi) #hier schon ändern? Vergleich einlesen Max, Einlesen meins?
csc = flatten_2D_array_manually(hyd.qs) #
cgc = flatten_2D_array_manually(hyd.qg) #
cwc = flatten_2D_array_manually(hyd.qc) #
crc = flatten_2D_array_manually(hyd.qr) #
q = flatten_2D_array_manually(hyd.qv)

rh=100*meteoSI.q2rh(q,t,p)
#rh = 26.3 * p * q * 1/(np.exp((17.67*(t-273.16))/(t-29.65)))/10000
print("2D arrays are flattened.")
#%%
'''
# combine all atmospheric composition parameters to one array for saving
atm_comp = np.zeros([len(time),56,4])
#atm_comp[:,:,0] = np.full([len(time),150],height)[:,:]
atm_comp[:,:,0] = z[:,:]
atm_comp[:,:,1] = p[:,:]
atm_comp[:,:,2] = t[:,:]
atm_comp[:,:,3] = rh[:,:]
'''
# set all hydrometeor nan-values to zero
cic[np.isnan(cic)]=0.
csc[np.isnan(csc)]=0.
cgc[np.isnan(cgc)]=0.
cwc[np.isnan(cwc)]=0.
crc[np.isnan(crc)]=0.

# create array of pamtra vertical output levels
flight_levels = np.arange(12000,14500,500) #vorher 7000 als start

# combine all hydrometeor arrays to one array (as needed by pamtra)
hydro_cmpl = np.zeros([len(time),56,5])
hydro_cmpl[:,:,0] = cwc[:,:] #specific cloud water content
hydro_cmpl[:,:,1] = cic[:,:] # specific cloud ice content
hydro_cmpl[:,:,2] = crc[:,:] # specific rain content
hydro_cmpl[:,:,3] = csc[:,:] #specific cloud snow content
hydro_cmpl[:,:,4] = cgc[:,:] #specific cloud graupel content

lons = flatten_1D_array_manually(grid.clat)
lats = flatten_1D_array_manually(grid.clon)

print("start of pamtra")
# PAMTRA SIMULATION

# create pamtra dictionairy containing all model input data for the pamtra simulation
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
#np.save('/home/u/u301032/orcestra/NN_IWP_retrieval/NN_training_and_development/pamData_test_factor_10.npy', pamData) 


# Load
#read_dictionary = np.load('/home/u/u301032/orcestra/NN_IWP_retrieval/NN_training_and_development/pamData.npy',allow_pickle='TRUE').item()

# define pamtra descriptorfile
# (defines physical interaction of radiation with cloud hydrometeors) 
descriptorFile1mom = np.array([ 
    # TODO to be reviewed, coefficients for m-D and v-D are changing
    #['hydro_name' 'as_ratio' 'liq_ice' 'rho_ms' 'a_ms' 'b_ms' 'alpha_as' 'beta_as' 'moment_in' 'nbin' 'dist_name' 'p_1' 'p_2' 'p_3' 'p_4' 'd_1' 'd_2' 'scat_name' 'vel_size_mod' 'canting']
    ('cwc_q', 1.0,  1, -99.0,   -99.0, -99.0,  -99.0, -99.0,  3,  1,   'mono',           -99.0, -99.0, -99.0, -99.0,  2.0e-5,  -99.0, 'mie-sphere', 'corPowerLaw_24388657.6_2.0', -99.0),
    ('iwc_q', 0.2, -1, -99.0,   130.0,   3.0,  0.684,   2.0,  3,  1,   'mono_cosmo_ice', -99.0, -99.0, -99.0, -99.0,   -99.0,  -99.0, 'ssrg-rt3_0.18_0.89_2.06_0.08', 'corPowerLaw_30.606_0.5533', -99.0),
    ('rwc_q', 1.0,  1, -99.0,   -99.0, -99.0,  -99.0, -99.0,  3,  100, 'exp',            -99.0, -99.0, 8.0e6, -99.0,  1.2e-4, 6.0e-3, 'mie-sphere', 'corPowerLaw_130.0_0.5', -99.0),
    ('swc_q', 0.6, -1, -99.0,   0.038,   2.0, 0.3971,  1.88,  3,  100, 'exp_cosmo_snow', -99.0, -99.0, -99.0, -99.0, 5.1e-11, 1.0e-2, 'ssrg-rt3_0.25_1.00_1.66_0.04', 'corPowerLaw_4.9_0.25', -99.0),
    ('gwc_q', 1.0, -1, -99.0,   169.6,   3.1,  -99.0, -99.0,  3,  100, 'exp',            -99.0, -99.0, 4.0e6, -99.0, 1.0e-10, 1.0e-2, 'mie-sphere', 'corPowerLaw_406.67_0.85', -99.0)
    ],)

# set up pamtra simulation
pam = pyPamtra.pyPamtra()
for df in descriptorFile1mom:
    pam.df.addHydrometeor(df)
#pam.df.readFile(descriptorfile)
pam.nmlSet['active'] = False
pam.nmlSet['passive'] = True
pam.set["pyVerbose"] = 0 
pam.p["noutlevels"] = len(flight_levels)
pam.createProfile(**pamData)

# calculate integrated values of hydrometeors
pam.addIntegratedValues()



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
# RUN & SAVE pamtra simulation

# select frequencies to be simulated
freqs = get_HAMP_freqs_of('all_2side')

# run pamtra simulation
pam.runParallelPamtra(
    freqs, 
    pp_deltaX=1, 
    pp_deltaY=1, 
    pp_deltaF=1, 
    pp_local_workers="auto")

# save pamtra simulation results to netcdf file
pam.writeResultsToNetCDF(
    f'/work/um0203/u301032/PAMTRA_output/PAMTRA-ICON_{DATE}_{output_name}.nc',
    xarrayCompatibleOutput=True)
#/work/um0203/u301238/PAMTRA/PAMTRA_NN_training_data/PAMTRA-ICON_{DATE}_4000rndm-profiles_all_hamp_freqs_v4.nc
# save integrated values of hydrometeors and water vapor to numpy array
bulk_values = np.concatenate((np.squeeze(pam.p['hydro_wp']),pam.p['iwv']),axis=1).shape
#np.save(f'/work/um0203/u301238/PAMTRA/PAMTRA_NN_training_data/PAMTRA-ICON_{DATE}_4000rndm-profiles_bulk_values_v5',bulk_values)
np.save(f'/work/um0203/u301032/PAMTRA_output/PAMTRA-ICON_{DATE}_{output_name}_bulk_values_v5',bulk_values)

print("")
print("*** PAMTRA simulation finished and netcdf files saved ***")
print("")
# %%
