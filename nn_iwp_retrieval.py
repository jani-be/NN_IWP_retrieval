import numpy as np
import xarray as xr
import src
from tensorflow import keras

# path to directory containing the trained neural networks for all altitudes
PATH_to_NNs = '/Users/maxringel/Documents/Studium/Studium_MSc/master_thesis/code/Code_levante_backup/NN_IWP_retrieval/NNs'

def standardize_nn_input(TBs, nn_level=12500,version=1):
    """
    Function for standardizing every TB of nn input array by subtracting 
    the mean TB and divding by std of TBs of corresponding nn training data.

    IMPORTANT: Always standardize by the training data of the corresponding 
    neural network!
    """
    
    TBs = np.asarray(TBs)

    # if single TB observation provided extend dims to 2
    if len(TBs.shape) == 1:
        TBs = TBs[np.newaxis,:]

    # read in corresponding training data of given neural network
    TBs_train = xr.open_dataset(PATH_to_NNs+f'/train_test_data/train_test_data_nn_24-32-1_reg_{nn_level}m_v{version}.nc').tb_train.values
    
    if TBs.shape[1] != TBs_train.shape[1]:
        print("Given TB array has to have same frequencies as provided in the respective training database of the NN.")
        return 
    
    # standardize every TB value of array
    TBs_centered = np.zeros(TBs.shape)
    for channel in range(TBs.shape[1]):

        mu_train = np.nanmean(TBs_train[:,channel])
        sigma_train = np.nanstd(TBs_train[:,channel])   

        TBs_centered[:,channel] = (TBs[:,channel] - mu_train)/sigma_train
            
    return TBs_centered

def retrieve_IWP(TBs,altitude):
    """
    Function to retrieve the IWP from a given set of brightness temperatures and their corresponding
    measurement altitudes. 

    IMPORTANT: 
    
    - These neural networks are trained only for the input of all HAMP frequencies, excluding
    the 2nd G-band channel (183.31±1.4GHz). The input TB-array has to be arranged in the following frequency order:
        array(
            [ 22.24,  23.04,  23.84,  25.44,  26.24,  27.84,  31.4 ,  50.3 ,
            51.76,  52.8 ,  53.75,  54.94,  56.66,  58.  ,  90.  , 120.15,
            121.05, 122.95, 127.25, 183.91, 185.81, 186.81, 188.31, 190.81]
            )
    Every other choice of HAMP channels or arrangement will lead to erroneous results!

    - Only measurements which are conducted within 8000-13500m altitude should be given as the networks are trained
    for this range. Retrieval results for measurements of other (especially lower) altitudes cannot be trusted.

    Note: Best results are obtained for TBs which are already corrected in terms of measurement offsets etc.!
    """

    # Load all trained neural networks of all alltitudes. Three versions exist for every altitude layer.
    NN_13000_V1 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_13000m_v1',compile=False)
    NN_13000_V2 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_13000m_v2',compile=False)
    NN_13000_V3 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_13000m_v3',compile=False)
    
    NN_12500_V1 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_12500m_v1',compile=False)
    NN_12500_V2 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_12500m_v2',compile=False)
    NN_12500_V3 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_12500m_v3',compile=False)
    
    NN_12000_V1 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_12000m_v1',compile=False)
    NN_12000_V2 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_12000m_v2',compile=False)
    NN_12000_V3 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_12000m_v3',compile=False)
    
    NN_11500_V1 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_11500m_v1',compile=False)
    NN_11500_V2 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_11500m_v2',compile=False)
    NN_11500_V3 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_11500m_v3',compile=False)
    
    NN_11000_V1 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_11000m_v1',compile=False)
    NN_11000_V2 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_11000m_v2',compile=False)
    NN_11000_V3 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_11000m_v3',compile=False)
    
    NN_10500_V1 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_10500m_v1',compile=False)
    NN_10500_V2 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_10500m_v2',compile=False)
    NN_10500_V3 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_10500m_v3',compile=False)
    
    NN_10000_V1 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_10000m_v1',compile=False)
    NN_10000_V2 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_10000m_v2',compile=False)
    NN_10000_V3 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_10000m_v3',compile=False)
    
    NN_9500_V1 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_9500m_v1',compile=False)
    NN_9500_V2 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_9500m_v2',compile=False)
    NN_9500_V3 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_9500m_v3',compile=False)
    
    NN_9000_V1 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_9000m_v1',compile=False)
    NN_9000_V2 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_9000m_v2',compile=False)
    NN_9000_V3 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_9000m_v3',compile=False)
    
    NN_8500_V1 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_8500m_v1',compile=False)
    NN_8500_V2 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_8500m_v2',compile=False)
    NN_8500_V3 = keras.models.load_model(PATH_to_NNs+'/dnn_model_iwp_24-32-1_reg_8500m_v3',compile=False)
    
    # Create array of altitude levels on which neural networks have been trained
    NN_levels = np.arange(8500,13500,500)
    # Sort every measurement to respective neural-network altitude layer (given by NN_levels),
    # according to its observation altitude
    levels = np.array([src.find_nearest_value(NN_levels,altitude[i]) for i in range(len(altitude))])
    
    # Allocate ouput arrays for IWP prediction of every nn version
    IWP_V1 = np.zeros(len(TBs))
    IWP_V1[:] = np.nan
    IWP_V2 = np.zeros(len(TBs))
    IWP_V2[:] = np.nan
    IWP_V3 = np.zeros(len(TBs))
    IWP_V3[:] = np.nan

    print("")
    print("Retrieving IWP from HAMP TBs...")
    print("")

    # Predict IWP from TBs on every altitude layer which contains observations of TBs
    # Before predicting, every TB observation is standardized according to the function above
    if len(levels[levels==13000]) != 0:
        print(f"Retrieving from 13000m ({len(levels[levels==13000])} TBs)")
        IWP_V1[np.where(levels==13000)] = NN_13000_V1.predict(standardize_nn_input(TBs[levels==13000],nn_level=13000,version=1))[:,0]**2
        IWP_V2[np.where(levels==13000)] = NN_13000_V2.predict(standardize_nn_input(TBs[levels==13000],nn_level=13000,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==13000)] = NN_13000_V3.predict(standardize_nn_input(TBs[levels==13000],nn_level=13000,version=3),verbose=0)[:,0]**2
    if len(levels[levels==12500]) != 0:
        print(f"Retrieving from 12500m ({len(levels[levels==12500])} TBs)")
        IWP_V1[np.where(levels==12500)] = NN_12500_V1.predict(standardize_nn_input(TBs[levels==12500],nn_level=12500,version=1))[:,0]**2
        IWP_V2[np.where(levels==12500)] = NN_12500_V2.predict(standardize_nn_input(TBs[levels==12500],nn_level=12500,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==12500)] = NN_12500_V3.predict(standardize_nn_input(TBs[levels==12500],nn_level=12500,version=3),verbose=0)[:,0]**2
    if len(levels[levels==12000]) != 0:
        print(f"Retrieving from 12000m ({len(levels[levels==12000])} TBs)")
        IWP_V1[np.where(levels==12000)] = NN_12000_V1.predict(standardize_nn_input(TBs[levels==12000],nn_level=12000,version=1))[:,0]**2
        IWP_V2[np.where(levels==12000)] = NN_12000_V2.predict(standardize_nn_input(TBs[levels==12000],nn_level=12000,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==12000)] = NN_12000_V3.predict(standardize_nn_input(TBs[levels==12000],nn_level=12000,version=3),verbose=0)[:,0]**2
    if len(levels[levels==11500]) != 0:
        print(f"Retrieving from 11500m ({len(levels[levels==11500])} TBs)")
        IWP_V1[np.where(levels==11500)] = NN_11500_V1.predict(standardize_nn_input(TBs[levels==11500],nn_level=11500,version=1))[:,0]**2
        IWP_V2[np.where(levels==11500)] = NN_11500_V2.predict(standardize_nn_input(TBs[levels==11500],nn_level=11500,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==11500)] = NN_11500_V3.predict(standardize_nn_input(TBs[levels==11500],nn_level=11500,version=3),verbose=0)[:,0]**2
    if len(levels[levels==11000]) != 0:
        print(f"Retrieving from 11000m ({len(levels[levels==11000])} TBs)")
        IWP_V1[np.where(levels==11000)] = NN_11000_V1.predict(standardize_nn_input(TBs[levels==11000],nn_level=11000,version=1))[:,0]**2
        IWP_V2[np.where(levels==11000)] = NN_11000_V2.predict(standardize_nn_input(TBs[levels==11000],nn_level=11000,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==11000)] = NN_11000_V3.predict(standardize_nn_input(TBs[levels==11000],nn_level=11000,version=3),verbose=0)[:,0]**2
    if len(levels[levels==10500]) != 0:
        print(f"Retrieving from 10500m ({len(levels[levels==10500])} TBs)")
        IWP_V1[np.where(levels==10500)] = NN_10500_V1.predict(standardize_nn_input(TBs[levels==10500],nn_level=10500,version=1))[:,0]**2
        IWP_V2[np.where(levels==10500)] = NN_10500_V2.predict(standardize_nn_input(TBs[levels==10500],nn_level=10500,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==10500)] = NN_10500_V3.predict(standardize_nn_input(TBs[levels==10500],nn_level=10500,version=3),verbose=0)[:,0]**2
    if len(levels[levels==10000]) != 0:
        print(f"Retrieving from 10000m ({len(levels[levels==10000])} TBs)")
        IWP_V1[np.where(levels==10000)] = NN_10000_V1.predict(standardize_nn_input(TBs[levels==10000],nn_level=10000,version=1))[:,0]**2
        IWP_V2[np.where(levels==10000)] = NN_10000_V2.predict(standardize_nn_input(TBs[levels==10000],nn_level=10000,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==10000)] = NN_10000_V3.predict(standardize_nn_input(TBs[levels==10000],nn_level=10000,version=3),verbose=0)[:,0]**2
    if len(levels[levels==9500]) != 0:
        print(f"Retrieving from 9500m ({len(levels[levels==9500])} TBs)")
        IWP_V1[np.where(levels==9500)] = NN_9500_V1.predict(standardize_nn_input(TBs[levels==9500],nn_level=9500,version=1))[:,0]**2    
        IWP_V2[np.where(levels==9500)] = NN_9500_V2.predict(standardize_nn_input(TBs[levels==9500],nn_level=9500,version=2),verbose=0)[:,0]**2    
        IWP_V3[np.where(levels==9500)] = NN_9500_V3.predict(standardize_nn_input(TBs[levels==9500],nn_level=9500,version=3),verbose=0)[:,0]**2    
    if len(levels[levels==9000]) != 0:
        print(f"Retrieving from 9000m ({len(levels[levels==9000])} TBs)")
        IWP_V1[np.where(levels==9000)] = NN_9000_V1.predict(standardize_nn_input(TBs[levels==9000],nn_level=9000,version=1))[:,0]**2
        IWP_V2[np.where(levels==9000)] = NN_9000_V2.predict(standardize_nn_input(TBs[levels==9000],nn_level=9000,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==9000)] = NN_9000_V3.predict(standardize_nn_input(TBs[levels==9000],nn_level=9000,version=3),verbose=0)[:,0]**2
    if len(levels[levels==8500]) != 0:
        print(f"Retrieving from 8500m ({len(levels[levels==8500])} TBs)")
        IWP_V1[np.where(levels==8500)] = NN_8500_V1.predict(standardize_nn_input(TBs[levels==8500],nn_level=8500,version=1))[:,0]**2
        IWP_V2[np.where(levels==8500)] = NN_8500_V2.predict(standardize_nn_input(TBs[levels==8500],nn_level=8500,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==8500)] = NN_8500_V3.predict(standardize_nn_input(TBs[levels==8500],nn_level=8500,version=3),verbose=0)[:,0]**2
    
    # Average over the IWP predictions of all three nn versions
    IWP = np.nanmean(np.array([IWP_V1,IWP_V2,IWP_V3]),axis=0)
    
    # Indicate the fraction of negative predictions (possible as the sqrt of IWP is retrieved).
    print("")
    print((len(IWP[IWP<0])/len(IWP)),"% negative predictions (=clipped to 0).")
    print("")
    # If negative IWP values exist, they are clipped to 0.
    IWP[IWP<0] = 0.

    return IWP, levels