import pyPamtra
import numpy as np
import xarray as xr
import sys
sys.path.append('/home/u/u301238/master_thesis/')
import src

RF_FLIGHTS = [
    #'RF02', #'20220312'
    #'RF03', #'20220313'
    #'RF04', #'20220314'
    #'RF05', #'20220315'
    #'RF06', #'20220316'*
    'RF07', #'20220320'
    #'RF08', #'20220321'
    #'RF09', #'20220328'
    #'RF10', #'20220329'
    #'RF11', #'20220330'
    #'RF12', #'20220401'
    #'RF13', #'20220404'
    #'RF14', #'20220407'
    #'RF15', #'20220408'
    #'RF16', #'20220410'
    #'RF17', #'20220411'
]
# * (data only as 24h forecast from March 15 available)

for FLIGHT_NR in RF_FLIGHTS:
    
    DATE = src.get_RF_date_of(FLIGHT_NR)
    
    print("")
    print(f"*** PAMTRA simulation for {FLIGHT_NR} on {DATE} ***")
    
    # READ IN ICON SURFACE & PROFILE DATA
    
    # Read in ICON file, containing the 4000 randomly selected ICON profiles of given date
    ICON_input = xr.open_dataset(f'/work/um0203/u301238/ICON/ICON_NN_training_data/ICON_NWP_HALOAC3-domain_{DATE}_4000rndm-profiles_v3.nc')
    
    ICON_input = ICON_input.update(
        {'time': ('time',src.unix_to_dt64_timestamps(ICON_input.time.values))})
    
    # convert datetime64 timestamps to numerical unix timestamps
    dt_time = ICON_input.time.values
    unix_time = np.array([
        dt_time[i].astype('datetime64[s]').astype('int') 
        for i in range(len(dt_time))])
    
    time = np.zeros([4000])
    time[0:1000] = unix_time[0]
    time[1000:2000] = unix_time[1]
    time[2000:3000] = unix_time[2]
    time[3000:4000] = unix_time[3]
    
    # multiply lengths of first two dimensions (time and ncells) for 1D reshaping
    reshaped_2D_to_1D = len(ICON_input.time)*len(ICON_input.ncells)
    
    # longitudes and latitudes of ICON profiles
    
    lons = src.flatten_1D_array_manually(ICON_input.lon)
    lats = src.flatten_1D_array_manually(ICON_input.lat)
    
    # create 1D time array containing time for all ICON profiles 
    #time = src.flatten_1D_array_manually(ICON_input.time)
    
    #ind = 0
    #for i in range(len(ICON_input.time)):
    #    time[ind:ind+len(ICON_input.ncells)] = ICON_input.time.values[i]
    #    ind = ind+len(ICON_input.ncells)

    # ICON surface properties of profiles
    t_g = src.flatten_1D_array_manually(ICON_input.t_g)
    sst = src.flatten_1D_array_manually(ICON_input.sst)
    u10 = src.flatten_1D_array_manually(ICON_input.u10)
    v10 = src.flatten_1D_array_manually(ICON_input.v10)
    fr_land = src.flatten_1D_array_manually(ICON_input.fr_land)
    fr_seaice = src.flatten_1D_array_manually(ICON_input.fr_seaice)

    # combine all atmospheric composition parameters to one array for saving
    surface = np.zeros([len(time),6])
    #surface[:,:,0] = np.full([len(time),150],height)[:,:]
    surface[:,0] = t_g[:]
    surface[:,1] = sst[:]
    surface[:,2] = u10[:]
    surface[:,3] = v10[:]
    surface[:,4] = fr_land[:]
    surface[:,5] = fr_seaice[:]

    # ICON profiles
    #height = ICON_input.height.values
    z = src.flatten_2D_array_manually(ICON_input.z)
    p = src.flatten_2D_array_manually(ICON_input.p)
    t = src.flatten_2D_array_manually(ICON_input.t)
    rh = src.flatten_2D_array_manually(ICON_input.rh)
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
    rh[np.isnan(rh)]=0.
    cic[np.isnan(cic)]=0.
    csc[np.isnan(csc)]=0.
    cgc[np.isnan(cgc)]=0.
    cwc[np.isnan(cwc)]=0.
    crc[np.isnan(crc)]=0.

    # create array of pamtra vertical output levels
    flight_levels = np.arange(7000,14500,500)

    # combine all hydrometeor arrays to one array (as needed by pamtra)
    hydro_cmpl = np.zeros([len(time),150,5])
    hydro_cmpl[:,:,0] = cwc[:,:]
    hydro_cmpl[:,:,1] = cic[:,:]
    hydro_cmpl[:,:,2] = crc[:,:]
    hydro_cmpl[:,:,3] = csc[:,:]
    hydro_cmpl[:,:,4] = cgc[:,:]

    # combine hydrometeors and profiles of z,t,p,rh to one array
    profiles = np.concatenate((atm_comp,hydro_cmpl),axis=2)

    # save icon profile and surface data directly as np.array for nn training purposes
    #np.save(f'/work/um0203/u301238/ICON/ICON_NN_training_data/ICON_NWP_HALOAC3-domain_{DATE}_4000rndm-profiles_profiles',profiles)
    #np.save(f'/work/um0203/u301238/ICON/ICON_NN_training_data/ICON_NWP_HALOAC3-domain_{DATE}_4000rndm-profiles_surface',surface)
    
    # PAMTRA SIMULATION
    
    # create pamtra dictionairy containing all model input data for the pamtra simulation
    pamData = dict()

    # time and location
    pamData["timestamp"] = time[:]
    pamData["lat"] = lats[:]
    pamData["lon"] = lons[:]

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
    pamData["hgt"] = z[:,:]
    pamData["press"] = p[:,:]
    pamData["temp"] = t[:,:]
    pamData["relhum"] = rh[:,:]
    pamData["hydro_q"] = hydro_cmpl[:,:,:]

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