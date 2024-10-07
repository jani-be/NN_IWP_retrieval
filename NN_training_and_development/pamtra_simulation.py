#%%

#import pyPamtra
import numpy as np
import xarray as xr
from netCDF4 import Dataset
#import sys
#sys.path.append('/home/u/u301238/master_thesis/')
#import src

#%%

# file paths

path = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/orcestra_1250m_0910/"
hydrometeor_file = path + "orcestra_1250m_0910_atm_3d_hydrometeors_DOM01_20240910T000000Z.nc"
thermodyn_file = path + "orcestra_1250m_0910_atm_3d_thermodynamics_DOM01_20240910T000000Z.nc"
meshdir = "/work/mh0492/m301067/orcestra/auxiliary-files/grids/"
meshname = "ORCESTRA_1250m_DOM01"

# reading files
grid = xr.open_dataset(meshdir+meshname+".nc")
hyd = xr.open_dataset(hydrometeor_file)
thermodyn =xr.open_dataset(thermodyn_file)
print(hyd,thermodyn)
print(grid)
# %%%

cic = src.flatten_2D_array_manually(ICON_input.cic)
csc = src.flatten_2D_array_manually(ICON_input.csc)
cgc = src.flatten_2D_array_manually(ICON_input.cgc)
cwc = src.flatten_2D_array_manually(ICON_input.cwc)
crc = src.flatten_2D_array_manually(ICON_input.crc)

# combine all atmospheric composition parameters to one array for saving
atm_comp = np.zeros([len(time),150,4])
#atm_comp[:,:,0] = np.full([len(time),150],height)[:,:]
atm_comp[:,:,0] = z[:,:]
atm_comp[:,:,1] = p[:,:]
atm_comp[:,:,2] = t[:,:]
atm_comp[:,:,3] = rh[:,:]

# set all hydrometeor nan-values to zero
cic[np.isnan(cic)]=0.
csc[np.isnan(csc)]=0.
cgc[np.isnan(cgc)]=0.
cwc[np.isnan(cwc)]=0.
crc[np.isnan(crc)]=0.

# create array of pamtra vertical output levels
flight_levels = np.arange(7000,14500,500)

# combine all hydrometeor arrays to one array (as needed by pamtra)
hydro_cmpl = np.zeros([len(time),150,5])
hydro_cmpl[:,:,0] = hyd.qc #cwc[:,:] #specific cloud water content
hydro_cmpl[:,:,1] = hyd.qi #cic[:,:] # specific cloud ice content
hydro_cmpl[:,:,2] = hyd. #crc[:,:]
hydro_cmpl[:,:,3] = hyd.qs #csc[:,:] #specific cloud snow content
hydro_cmpl[:,:,4] = hyd.qg #cgc[:,:] #specific cloud graupel content


# PAMTRA SIMULATION

# create pamtra dictionairy containing all model input data for the pamtra simulation
pamData = dict()

# time and location
pamData["timestamp"] = hyd.time #time[:]
pamData["lat"] = grid.clat #lats[:]
pamData["lon"] = grid.clon #lons[:]

# surface properties
pamData["groundtemp"] = t_g[:]
pamData["sfc_slf"] = fr_land[:]
pamData["sfc_sif"] = fr_seaice[:]
pamData["wind10u"] = u10[:]
pamData["wind10v"] = v10[:]
pamData["sfc_type"] = np.around(pamData['sfc_slf'])[:]
pamData["sfc_model"] = np.zeros(pamData['groundtemp'].shape)[:]
pamData["sfc_refl"]  = np.chararray(pamData['groundtemp'].shape)[:]
pamData["sfc_refl"][:] = 'S' # land  'F' # ocean 'L' lambertian, land
pamData["sfc_type"][(pamData['sfc_type'] == 0) & (pamData['sfc_sif'] > 0)] = 1

# vertical profiles
#pamData["hgt"] = np.array([height,]*len(time))[:,:]
pamData["hgt"] = hyd.height #z[:,:]
pamData["press"] = thermodyn.pfull #p[:,:]
pamData["temp"] = thermodyn.ta #t[:,:]
pamData["relhum"] =  #rh[:,:] # where to find ? 
pamData["hydro_q"] = #hydro_cmpl[:,:,:]

pamData["obs_height"] = np.zeros([len(time),1,len(flight_levels)])
pamData["obs_height"][:,:,:] = flight_levels
#pamData["obs_height"] = np.full([len(time),1],12500.)[:,:]

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

# RUN & SAVE pamtra simulation

# select frequencies to be simulated
freqs = src.get_HAMP_freqs_of('all_2side')

# run pamtra simulation
pam.runParallelPamtra(
    freqs, 
    pp_deltaX=1, 
    pp_deltaY=1, 
    pp_deltaF=1, 
    pp_local_workers="auto")

# save pamtra simulation results to netcdf file
pam.writeResultsToNetCDF(
    f'/work/um0203/u301238/PAMTRA/PAMTRA_NN_training_data/PAMTRA-ICON_{DATE}_4000rndm-profiles_all_hamp_freqs_v4.nc',
    xarrayCompatibleOutput=True)

# save integrated values of hydrometeors and water vapor to numpy array
bulk_values = np.concatenate((np.squeeze(pam.p['hydro_wp']),pam.p['iwv']),axis=1).shape
np.save(f'/work/um0203/u301238/PAMTRA/PAMTRA_NN_training_data/PAMTRA-ICON_{DATE}_4000rndm-profiles_bulk_values_v5',bulk_values)

print("")
print("*** PAMTRA simulation finished and netcdf files saved ***")
print("")