import numpy as np
from glob import glob
import xarray as xr

def find_nearest_value(array, value):
    """
    Function to return the value of the element of a given array which
    is closest to the given value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_RF_date_of(RF):
    """
    Function to return date of specified HALO-AC3 research flight.
    """
    
    RF_dates = {'RF02':'20220312',
                'RF03':'20220313',
                'RF04':'20220314',
                'RF05':'20220315',
                'RF06':'20220316',
                'RF07':'20220320',
                'RF08':'20220321',
                'RF09':'20220328',
                'RF10':'20220329',
                'RF11':'20220330',
                'RF12':'20220401',
                'RF13':'20220404',
                'RF14':'20220407',
                'RF15':'20220408',
                'RF16':'20220410',
                'RF17':'20220411',
                'RF18':'20220412',
               }
    
    return RF_dates[RF]


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

def load_all_hamp_obs(path,freqs='NN_freqs',sfc_type='all',alt_range=None,timestamps=None):
    """
    Function to load all HAMP observations during HALO-AC3, including surface mask.
    """

    # list of all HAMP radiometer files
    hamp_files = sorted(glob(path+'/radiometer_*_v1.6_TBcorr.nc'))

    # open as one dataset
    hamp = xr.open_mfdataset(
        hamp_files,
        combine='nested',
        concat_dim='time')
    
    # if altitude range is given, select only observations within this range
    if alt_range is not None:
        hamp = hamp.where((hamp.alt>=alt_range[0])&(hamp.alt<=alt_range[1]),drop=True)
    
    # if timestamps are given, select only observations during this timestamps
    if timestamps is not None:
        hamp = hamp.sel(time=timestamps,drop=True)
    
    # select observations only at given frequencies
    frequencies = get_HAMP_freqs_of(freqs)
    if (freqs == 'all_1side') or (freqs == 'G_band_1side'):
        frequencies = np.delete(frequencies,-1) # remove 195 GHz channel (was not working during HALO-AC3)
        
    hamp = hamp.sel(uniRadiometer_freq=xr.DataArray(frequencies,dims='uniRadiometer_freq'))

    # get surface mask of HAMP
    sfc_mask = hamp.surface_mask

    # select observations only over specified surface type
    if sfc_type == 'ocean':
        hamp = hamp.where(hamp.surface_mask == 0.,
                          #drop=True
                          )
    if sfc_type == 'land':
        hamp = hamp.where(hamp.surface_mask < 0.,
                          #drop=True
                          )
    if sfc_type == 'sea-ice':
        hamp = hamp.where(hamp.surface_mask > 0., 
                          #drop=True
                          )
    if sfc_type == 'all':
        hamp = hamp
        
    return hamp, sfc_mask

def mask_seaice_edges(TBs, sfc_mask, timerange=150):
    """
    Function to mask out observations close to sea-ice edges. 
    Timerange (in seconds) is the range within observations are masked
    before and/or after the first/last sea-ice contamination.

    * Could be improved a lot by finding a work-around for the loop *
    """
   
    mask_range = int(timerange*2)
    for i in range(len(sfc_mask)):
        
        if sfc_mask[i] != 0.:
            
            if i <= mask_range:
                TBs[:mask_range,:] = np.nan
            if i > mask_range:   
                TBs[(i-(int(mask_range/2))):(i+(int(mask_range/2))),:] = np.nan

    return TBs