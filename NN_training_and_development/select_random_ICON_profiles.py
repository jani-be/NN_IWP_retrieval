import xarray as xr
from glob import glob
import numpy as np
import scipy
import scipy.interpolate
import random
from netCDF4 import Dataset
import sys
sys.path.append('/home/u/u301238/master_thesis/')
import src

# v1: ICON profiles have been interpolated on regular vertical grid 
# v2: ICON profiles not interpolated (on original pressure levels)
# v3: Additionaly to v2 'qc' and 'qi' saved as non-diagnostically variables

RF_FLIGHTS = [
    'RF02', #'20220312'
    'RF03', #'20220313'
    'RF04', #'20220314'
    'RF05', #'20220315'
    'RF06', #'20220316'*
    'RF07', #'20220320'
    'RF08', #'20220321'
    'RF09', #'20220328'
    'RF10', #'20220329'
    'RF11', #'20220330'
    'RF12', #'20220401'
    'RF13', #'20220404'
    'RF14', #'20220407'
    'RF15', #'20220408'
    'RF16', #'20220410'
    'RF17', #'20220411'
    'RF18', #'20220412'
]
# * (data only as 24h forecast from March 15 available)

for RF in RF_FLIGHTS: 
    
    DATE = src.get_RF_date_of(RF)
    
    print("*** Profile selection for ",DATE," ***")
    
    #DATE = sys.argv[1]
    MONTH = DATE[4:6]
    DAY = DATE[6:9]
    
    if DATE == '20220315': 
        # create list of all ICON files (not *_vs directory in this case)
        ICON_files = sorted(glob('/work/bb1086/haloac3/icon_nwp/'
                                 f'2022-0{DATE[5:6]}-{DATE[6:8]}/cloud_DOM01_ML_{DATE}*'))
        
        # create list of all ICON_forcing files 
        # (contain surface properties) of specified date
        ICON_forcing_files = sorted(glob('/work/bb1086/haloac3/icon_nwp/'
                                     f'2022-0{DATE[5:6]}-{DATE[6:8]}/forcing_DOM01_ML_{DATE}*'))

    if DATE != '20220315': # create list of all ICON files of specified date
        ICON_files = sorted(glob('/work/bb1086/haloac3/icon_nwp/'
                                 f'2022-0{DATE[5:6]}-{DATE[6:8]}*/cloud_DOM01_ML_{DATE}*'))
        
        # create list of all ICON_forcing files 
        # (contain surface properties) of specified date
        ICON_forcing_files = sorted(glob('/work/bb1086/haloac3/icon_nwp/'
                                         f'2022-0{DATE[5:6]}-{DATE[6:8]}*/forcing_DOM01_ML_{DATE}*'))

    def preprocess_icon_files(ds):
        """
        Function to preprocess ICON files before merging them 
        into one single multifile with xr.open_mfdataset() by:
         - converting (numerical) ICON timestamps into datetime timestamps
         - droping all but the desired variables 
         - droping unnecessary height dimensions
        """
        ds = src.adapt_icon_time_index(ds,DATE)
        ds = src.xr_keep(ds, ['z_mc','pres',
                              'temp',
                              'rh','qv',
                              'qc','qr','qi','qs','qg',
                              'tqc_dia',
                              'tqi_dia',
                              'tqv_dia',
                              'tot_qc_dia','tot_qi_dia',
                             ])
        ds = ds.drop(['height_2','height_3','height_4'])
        return ds


    # read in all ICON files of specified date,
    # preprocess them and combine them to one single mfdataset
    ICON = xr.open_mfdataset(
        ICON_files,
        preprocess=preprocess_icon_files,
        combine='nested',
        concat_dim='time',
    )

    def preprocess_icon_forcing_files(ds):
        """
        Function to preprocess ICON files before merging them 
        into one single multifile with xr.open_mfdataset() by:
         - converting (numerical) ICON timestamps into datetime timestamps
         - droping unnecessary dimensions
         - droping unnecessary variables
        """
        ds = src.adapt_icon_time_index(ds,DATE)
        ds = ds.drop_dims(
            ['ncells','vertices_2','bnds',
             'height','height_2','height_3','height_4',
             'depth','depth_2'])
        ds = ds.drop(
            ['gz0','t_ice','h_ice','alb_si','w_i','t_snow','w_snow',
             'rho_snow','h_snow','snowfrac_lc','freshsnow'])

        return ds


    # read in all ICON forcing files of specified date,
    # preprocess them and combine them to one single mfdataset
    ICON_forcing = xr.open_mfdataset(
        ICON_forcing_files,
        preprocess=preprocess_icon_forcing_files,
        combine='nested',
        concat_dim='time')

    # create DataArray with desired timestamps of subselection
    temporal_sel = xr.DataArray([np.datetime64(f'2022-{MONTH}-{DAY}T00:00:00.000000000'),
                                 np.datetime64(f'2022-{MONTH}-{DAY}T06:00:00.000000000'),
                                 np.datetime64(f'2022-{MONTH}-{DAY}T12:00:00.000000000'),
                                 np.datetime64(f'2022-{MONTH}-{DAY}T18:00:00.000000000')],
                                dims=['time'])
    # respective labels for plotting
    labels = ['00 UTC','06 UTC','12 UTC','18 UTC']

    ICON = ICON.sel(time=temporal_sel)
    ICON_forcing = ICON_forcing.sel(time=temporal_sel)

    # specify how many profiles are selected for each timestamp
    SAMPLESIZE = 1000

    #%%
    print('')
    print('Allocating ICON input arrays (takes some time)...')
    # allocate ICON input arrays
    # (makes indexing inside loop faster than using xr.sel())
    
    # surface properties
    ICON_lons = np.array(ICON.clon.values[:]) # longitude of ICON grid cell
    ICON_lats = np.array(ICON.clat.values[:]) # latitude of ICON grid cell
    ICON_t_g = np.array(ICON_forcing.t_g.values[:,:]) # ground temperature
    ICON_sst = np.array(ICON_forcing.t_seasfc.values[:,:]) # sea surface temperature
    ICON_u10 = np.array(ICON_forcing.u_10m.values[:,0,:]) # u-wind at 10m
    ICON_v10 = np.array(ICON_forcing.v_10m.values[:,0,:]) # v-wind at 10m
    ICON_fr_seaice = np.array(ICON_forcing.fr_seaice.values[:,:]) #seaice fraction
    ICON_fr_land = np.array(ICON_forcing.fr_land.values[:,:]) #land fraction
    
    # profiles
    ICON_height = np.array(ICON.z_mc.values[:,:,:]) #geometric_height_at_full_level_center
    ICON_press = np.array(ICON.pres.values[:,:,:]) #pressure
    ICON_temp = np.array(ICON.temp.values[:,:,:]) #temperature
    ICON_relhum = np.array(ICON.rh.values[:,:,:]) #relativ humidity
    ICON_spechum = np.array(ICON.qv.values[:,:,:]) #specific humidity
    
    ICON_cwc = np.array(ICON.qc.values[:,:,:]) #specific cloud water content
    ICON_cwc_dia = np.array(ICON.tot_qc_dia.values[:,:,:]) #total specific cloud water content (diagnostic)
    ICON_crc = np.array(ICON.qr.values[:,:,:]) #rain mixing ratio
    ICON_cic = np.array(ICON.qi.values[:,:,:]) #specific cloud ice content
    ICON_cic_dia = np.array(ICON.tot_qi_dia.values[:,:,:]) #total specific cloud ice content (diagnostic)
    ICON_csc = np.array(ICON.qs.values[:,:,:]) #snow mixing ratio
    ICON_cgc = np.array(ICON.qg.values[:,:,:]) #specific graupel content
    
    ICON_cwp = np.array(ICON.tqc_dia.values[:,:]) #total column integrated cloud water (diagnostic)
    ICON_iwv = np.array(ICON.tqv_dia.values[:,:]) #total column integrated water wapor (diagnostic)
    ICON_cip = np.array(ICON.tqi_dia.values[:,:]) #total column integrated cloud ice (diagnostic) 
    print('ICON Input arrays allocated.')

    # allocate output arrays
    #surface properties
    lons = np.empty([len(temporal_sel),SAMPLESIZE])
    lats = np.empty([len(temporal_sel),SAMPLESIZE])
    t_g = np.empty([len(temporal_sel),SAMPLESIZE])
    sst = np.empty([len(temporal_sel),SAMPLESIZE])
    u10 = np.empty([len(temporal_sel),SAMPLESIZE])
    v10 = np.empty([len(temporal_sel),SAMPLESIZE])
    fr_land = np.empty([len(temporal_sel),SAMPLESIZE])
    fr_seaice = np.empty([len(temporal_sel),SAMPLESIZE])
    #proifile data
    z = np.empty([len(temporal_sel),SAMPLESIZE,150])
    p = np.empty([len(temporal_sel),SAMPLESIZE,150])
    temp = np.empty([len(temporal_sel),SAMPLESIZE,150])
    rh = np.empty([len(temporal_sel),SAMPLESIZE,150])
    q = np.empty([len(temporal_sel),SAMPLESIZE,150])
    cic = np.empty([len(temporal_sel),SAMPLESIZE,150])
    cic_dia = np.empty([len(temporal_sel),SAMPLESIZE,150])
    csc = np.empty([len(temporal_sel),SAMPLESIZE,150])
    cgc = np.empty([len(temporal_sel),SAMPLESIZE,150])
    cwc = np.empty([len(temporal_sel),SAMPLESIZE,150])
    cwc_dia = np.empty([len(temporal_sel),SAMPLESIZE,150])
    crc = np.empty([len(temporal_sel),SAMPLESIZE,150])
    cip = np.empty([len(temporal_sel),SAMPLESIZE])
    cwp = np.empty([len(temporal_sel),SAMPLESIZE])
    iwv = np.empty([len(temporal_sel),SAMPLESIZE])

    # define new (linear) height vector to interpolate ICON data onto
    #z_vector_new = np.linspace(0,15000,150)

    # loop through desired timestamps and choose a random selection 
    # of 3000 profiles for each timestamp. Then choose only the profiles 
    # over ocean.
    for t in range(len(temporal_sel)):
        
        # get all the cell indices of cells over ice-free ocean
        all_ocean_profiles = np.where((ICON_fr_seaice[t,:]==0)&(ICON_fr_land[t,:]==0))[0].tolist()
        # choose a random subsample of these cell indices of SAMPLESIZE
        rndm_ocean_profiles = random.sample(all_ocean_profiles,SAMPLESIZE)
        
        # get lons and lats of selected cells and convert to decimal degrees
        lons[t,:] = np.rad2deg(ICON_lons[rndm_ocean_profiles])
        lats[t,:] = np.rad2deg(ICON_lats[rndm_ocean_profiles])

        # loop through every selected profile
        # (loop probably not needed - realised later -.-)
        for j in range(SAMPLESIZE):

            t_g[t,j] = ICON_t_g[t,rndm_ocean_profiles[j]]
            sst[t,j] = ICON_sst[t,rndm_ocean_profiles[j]]
            u10[t,j] = ICON_u10[t,rndm_ocean_profiles[j]]
            v10[t,j] = ICON_v10[t,rndm_ocean_profiles[j]]
            fr_land[t,j] = ICON_fr_land[t,rndm_ocean_profiles[j]]
            fr_seaice[t,j] = ICON_fr_seaice[t,rndm_ocean_profiles[j]]

            z[t,j,:] = np.flip(ICON_height[t,:,rndm_ocean_profiles[j]])
            p[t,j,:] = np.flip(ICON_press[t,:,rndm_ocean_profiles[j]])
            temp[t,j,:] = np.flip(ICON_temp[t,:,rndm_ocean_profiles[j]])
            rh[t,j,:] = np.flip(ICON_relhum[t,:,rndm_ocean_profiles[j]])
            q[t,j,:] = np.flip(ICON_spechum[t,:,rndm_ocean_profiles[j]])
            cwc[t,j,:] = np.flip(ICON_cwc[t,:,rndm_ocean_profiles[j]])
            cwc_dia[t,j,:] = np.flip(ICON_cwc_dia[t,:,rndm_ocean_profiles[j]])
            crc[t,j,:] = np.flip(ICON_crc[t,:,rndm_ocean_profiles[j]])
            cic[t,j,:] = np.flip(ICON_cic[t,:,rndm_ocean_profiles[j]])
            cic_dia[t,j,:] = np.flip(ICON_cic_dia[t,:,rndm_ocean_profiles[j]])
            csc[t,j,:] = np.flip(ICON_csc[t,:,rndm_ocean_profiles[j]])
            cgc[t,j,:] = np.flip(ICON_cgc[t,:,rndm_ocean_profiles[j]])
            cwp[t,j] = ICON_cwp[t,rndm_ocean_profiles[j]]
            iwv[t,j] = ICON_iwv[t,rndm_ocean_profiles[j]]
            cip[t,j] = ICON_cip[t,rndm_ocean_profiles[j]]
            
            # interpolate ICON profile variables on new linear vertical grid
            """
            z_vector = np.flip(ICON_height[t,:,rndm_ocean_profiles[j]])

            interpolate_t = scipy.interpolate.interp1d(
                z_vector,temp[t,j,:],fill_value='extrapolate')
            interpolate_p = scipy.interpolate.interp1d(
                z_vector,p[t,j,:],fill_value='extrapolate')
            interpolate_rh = scipy.interpolate.interp1d(
                z_vector,rh[t,j,:],fill_value='extrapolate')
            interpolate_cic = scipy.interpolate.interp1d(
                z_vector,cic[t,j,:],fill_value='extrapolate')
            interpolate_csc = scipy.interpolate.interp1d(
                z_vector,csc[t,j,:],fill_value='extrapolate')
            interpolate_cgc = scipy.interpolate.interp1d(
                z_vector,cgc[t,j,:],fill_value='extrapolate')
            interpolate_cwc = scipy.interpolate.interp1d(
                z_vector,cwc[t,j,:],fill_value='extrapolate')
            interpolate_crc = scipy.interpolate.interp1d(
                z_vector,crc[t,j,:],fill_value='extrapolate')

            temp[t,j,:] = interpolate_t(z_vector_new)
            p[t,j,:] = interpolate_p(z_vector_new)
            rh[t,j,:] = interpolate_rh(z_vector_new)
            cic[t,j,:] = interpolate_cic(z_vector_new)
            csc[t,j,:] = interpolate_csc(z_vector_new)
            cgc[t,j,:] = interpolate_cgc(z_vector_new)
            cwc[t,j,:] = interpolate_cwc(z_vector_new)
            crc[t,j,:] = interpolate_crc(z_vector_new)
            """

        # set unrealistic values to NaN
        temp[temp<0]=np.nan
        p[p<0]=np.nan
        rh[rh<0]=np.nan
        rh[rh>100]=np.nan
        q[q<0]=np.nan
        cic[cic<0]=np.nan
        cic_dia[cic_dia<0]=np.nan
        csc[csc<0]=np.nan
        cgc[cgc<0]=np.nan
        cwc[cwc<0]=np.nan
        cwc_dia[cwc_dia<0]=np.nan
        crc[crc<0]=np.nan
        iwv[iwv<0]=np.nan

    # Saving ICON data to netcdf file
    filename = (
        '/work/um0203/u301238/ICON/ICON_NN_training_data/'
        f'ICON_NWP_HALOAC3-domain_{DATE}_{str(len(temporal_sel)*SAMPLESIZE)}rndm-profiles_v3.nc')
    ds = Dataset(filename,'w','NETCDF4')

    # time and height dimension
    ds.createDimension('time',len(temporal_sel))
    ds.createDimension('ncells',SAMPLESIZE)
    ds.createDimension('vlevel',150)

    # time
    time_var = ds.createVariable('time','float64',('time'))
    time_var[:] = (temporal_sel - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1,'s')
    # ncells
    ncells_var = ds.createVariable('ncells','int64',('ncells'))
    ncells_var.long_name = 'Index of gridcell'
    ncells_var[:] = np.arange(1,SAMPLESIZE+1)
    # vlevel 
    levelvar = ds.createVariable('vlevel','float64',('vlevel'))
    levelvar.long_name = 'Index of vertical level'
    levelvar[:] = np.arange(1.,151.)
    # height (new height vector)
    #heightvar = ds.createVariable('height','float64',('height'))
    #heightvar[:] = z_vector_new
    #heightvar.setncattr('units','m')
    # lons and lats
    lon_var = ds.createVariable('lon','float64',('time','ncells'))
    lat_var = ds.createVariable('lat','float64',('time','ncells'))
    lon_var.long_name = 'Longitude'
    lat_var.long_name = 'Latitude'
    lon_var.setncattr('units','degrees east')
    lat_var.setncattr('units','degrees north')
    lon_var[:,:] = lons
    lat_var[:,:] = lats
    # t_g
    t_g_var = ds.createVariable('t_g','float64',('time','ncells'))
    t_g_var.long_name = 'Ground temperature'
    t_g_var.setncattr('units','K')
    t_g_var[:] = t_g
    # sst
    sst_var = ds.createVariable('sst','float64',('time','ncells'))
    sst_var.long_name = 'Sea surface temperaure'
    sst_var.setncattr('units','K')
    sst_var[:] = sst
    # u10
    u10_var = ds.createVariable('u10','float64',('time','ncells'))
    u10_var.long_name = 'u-wind at 10m'
    u10_var.setncattr('units','m/s')
    u10_var[:] = u10
    # v10
    v10_var = ds.createVariable('v10','float64',('time','ncells'))
    v10_var.long_name = 'v-wind at 10m'
    v10_var.setncattr('units','m/s')
    v10_var[:] = v10
    # fr_land
    fr_land_var = ds.createVariable('fr_land','float64',('time','ncells'))
    fr_land_var.long_name = 'Fraction of land inside ICON grid cell'
    fr_land_var.setncattr('units','1')
    fr_land_var[:] = fr_land
    # fr_seaice
    fr_seaice_var = ds.createVariable('fr_seaice','float64',('time','ncells'))
    fr_seaice_var.long_name = 'Fraction of seaice inside ICON grid cell'
    fr_seaice_var.setncattr('units','1')
    fr_seaice_var[:] = fr_seaice
    # cip
    cip_var = ds.createVariable('cip','float64',('time','ncells'))
    cip_var.long_name = 'Cloud ice path (vertically integrated cic)'
    cip_var.setncattr('units','kg/m2')
    cip_var[:,:] = cip
    # cwp
    cwp_var = ds.createVariable('cwp','float64',('time','ncells'))
    cwp_var.long_name = 'Cloud water path (vertically integrated cwc)'
    cwp_var.setncattr('units','kg/m2')
    cwp_var[:,:] = cwp
    # iwv
    iwv_var = ds.createVariable('iwv','float64',('time','ncells'))
    iwv_var.long_name = 'Integrated water vapour (vertically integrated q)'
    iwv_var.setncattr('units','kg/m2')
    iwv_var[:,:] = iwv
    # temperature variable
    t_var = ds.createVariable('t','float64',('time','ncells','vlevel'))
    t_var.long_name = 'Temperature'
    t_var.setncattr('units','Kelvin')
    t_var[:,:,:] = temp
    # height variable
    z_var = ds.createVariable('z','float64',('time','ncells','vlevel'))
    z_var.long_name = 'Height'
    z_var.setncattr('units','m')
    z_var[:,:,:] = z
    # pressure variable
    p_var = ds.createVariable('p','float64',('time','ncells','vlevel'))
    p_var.long_name = 'Pressure'
    p_var.setncattr('units','Pa')
    p_var[:,:,:] = p
    # rel. humidity variable
    rh_var = ds.createVariable('rh','float64',('time','ncells','vlevel'))
    rh_var.long_name = 'Relativ humidity'
    rh_var.setncattr('units','%')
    rh_var[:,:,:] = rh
    # spe. humidity variable
    q_var = ds.createVariable('q','float64',('time','ncells','vlevel'))
    q_var.long_name = 'Specific humidity'
    q_var.setncattr('units','kg/kg')
    q_var[:,:,:] = q
    # cloud ice variable
    cic_var = ds.createVariable('cic','float64',('time','ncells','vlevel'))
    cic_var.long_name = 'Cloud ice content'
    cic_var.setncattr('units','kg/kg')
    cic_var[:,:,:] = cic
    # cloud ice variable (diagnostic)
    cic_dia_var = ds.createVariable('cic_dia','float64',('time','ncells','vlevel'))
    cic_dia_var.long_name = 'Cloud ice content (diagnostic)'
    cic_dia_var.setncattr('units','kg/kg')
    cic_dia_var[:,:,:] = cic_dia
    # cloud snow variable
    csc_var = ds.createVariable('csc','float64',('time','ncells','vlevel'))
    csc_var.long_name = 'Cloud snow content'
    csc_var.setncattr('units','kg/kg')
    csc_var[:,:,:] = csc
    # cloud graupel variable
    cgc_var = ds.createVariable('cgc','float64',('time','ncells','vlevel'))
    cgc_var.long_name = 'Cloud graupel content'
    cgc_var.setncattr('units','kg/kg')
    cgc_var[:,:,:] = cgc
    # cloud water variable
    cwc_var = ds.createVariable('cwc','float64',('time','ncells','vlevel'))
    cwc_var.long_name = 'Cloud water content'
    cwc_var.setncattr('units','kg/kg')
    cwc_var[:,:,:] = cwc
    # cloud water variable (diagnostic)
    cwc_dia_var = ds.createVariable('cwc_dia','float64',('time','ncells','vlevel'))
    cwc_dia_var.long_name = 'Cloud water content (diagnostic)'
    cwc_dia_var.setncattr('units','kg/kg')
    cwc_dia_var[:,:,:] = cwc_dia
    # cloud rain variable
    crc_var = ds.createVariable('crc','float64',('time','ncells','vlevel'))
    crc_var.long_name = 'Cloud rain content'
    crc_var.setncattr('units','kg/kg')
    crc_var[:,:,:] = crc

    ds.close()
    
    print('')
    print("*** Netcdf file saved ***")
    print('')