"""
Training of Neural Network

This script trains the Neural Network for all hights and all hydrometeors.

 Here one can enter the findings from from NN-development.ipnyb
 """


#%%
import tensorflow as tf
#import levenberg_marquardt as lm
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import seaborn as sns
from glob import glob
import xarray as xr
#import typhon as ty
import sys
#sys.path.append('/home/u/u301238/master_thesis/')
sys.path.append('/home/u/u301032/orcestra/NN_IWP_retrieval/NN_training_and_development/')
print(sys.path)
#import src

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


#%% PARAMETERS

variable = 'IWV' # 'IWP' # 'LWP' #
training_version = "v1"
name_pamtra_run = "cells_025x025_2h"#"all_area_1000th_cell" # 
flight_levels =[11400,12650,13000,13250,13600,13850,14450,15000	]






# General settings of the neural network
NR_of_NEURONS_L1 = 32
BIAS_INIT = 'zeros'
WEIGHT_INIT = 'random_normal'
ACTIVATION_HL = 'tanh'
ACTIVATION_OP = 'linear'
LOSS_FUNCTION = tf.keras.losses.MeanSquaredError()
LEARNING_RATE = 0.001

EPOCHS = 1000
BATCH_SIZE = 50



#%% LOAD PREPROCESSED TRANING DATA
data=[]
names=[ 'IWV', 'IWP', 'LWP','t_steps','cell_selection']
for i in range(len(names)):
    data.append(np.load('/work/um0203/u301032/master_thesis/ML_input/' + name_pamtra_run + '_' + names[i] + '.npy'))
IWV, IWP, LWP,t_steps,cell_selection =data
IWP=1000*IWP
LWP=1000*LWP
IWP[IWP<1]=0.
LWP[LWP<1]=0.
TBs_array=[]
mu_array=[]
sigma_array=[]

#for altitude in flight_levels:
#    TBs_array.append(np.load('/work/um0203/u301032/master_thesis/ML_input/' + name_pamtra_run + '_TBs_altitude_' + str(altitude) + 'm.npy'))






#%% Functions
### Standardizing Input
##### standardize all TBs along their respective channel as a normal distribution with mean 0 and std of 1
def standardize_nn_training_data(TBs_train):
        
    TBs_centered = np.zeros(TBs_train.shape)
    mu_train = np.zeros(TBs_train.shape[1])
    sigma_train = np.zeros(TBs_train.shape[1])
    for channel in range(TBs_train.shape[1]):

        mu_train[channel] = np.nanmean(TBs_train[:,channel])
        sigma_train[channel] = np.nanstd(TBs_train[:,channel])   

        TBs_centered[:,channel] = (TBs_train[:,channel] - mu_train[channel])/sigma_train[channel]
        
    return TBs_centered, mu_train, sigma_train
    
def standardize_nn_input_data_v2(TBs, mu, sigma):
    
    TBs = np.asarray(TBs)
    # if single TB observation provided extend dims to 2
    if len(TBs.shape) == 1:
        TBs = TBs[np.newaxis,:]
    
    TBs_centered = np.zeros(TBs.shape)
    for channel in range(TBs.shape[1]):

        TBs_centered[:,channel] = (TBs[:,channel] - mu[channel])/sigma[channel]
    
    return TBs_centered


def splitting_in_train_test_validate(A,t_steps,slices_train,slices_validate,slices_test):    
    # Splits in timesteps
    x=np.arange(0,A.shape[0]+1,A.shape[0]/len(t_steps))
    x=x.astype(int)
    indexes = np.concatenate([np.arange(x[i],x[j]) for i,j in slices_train])
    train=  A[indexes]
    
    indexes = np.concatenate([np.arange(x[i],x[j]) for i,j in slices_validate])
    validate=  A[indexes]
    
    indexes = np.concatenate([np.arange(x[i],x[j]) for i,j in slices_test])
    test=  A[indexes]
    
    return train,test,validate

a=np.arange(1,33,4)
l=[]
for i in range(len(a)):
    l.append([a[i],a[i]+1])
slices_validate=np.asarray(l)    
a=np.arange(3,33,4)
l=[]
for i in range(len(a)):
    l.append([a[i],a[i]+1])
slices_test=np.asarray(l)   
a=np.arange(0,33,2)
l=[]
for i in range(len(a)):
    l.append([a[i],a[i]+1])
slices_train=np.asarray(l)    


#%%
def array_for_variable(v):
    match v:
        case 'IWP':
            return IWP
        case 'IWV':
            return IWV
        case 'LWP':
            return LWP  
hyd = array_for_variable(variable)       

for altitude in flight_levels:

    train_hyd, test_hyd, validate_hyd = splitting_in_train_test_validate(hyd,t_steps,slices_train,slices_validate,slices_test)
    train_hyd=np.concatenate([train_hyd, validate_hyd])

    TBs=np.load('/work/um0203/u301032/master_thesis/ML_input/' + name_pamtra_run + '_TBs_altitude_' + str(altitude) + 'm.npy')
    ## Filter for unrealistic BTs missing and determine calculation of flight level
    TB_input_vector=TBs
    # Only necessary if one wants to exclude certain frequencies.
    TB_input_vector = np.concatenate((
            TB_input_vector[:,0:7], # K-Band
            TB_input_vector[:,7:14], # V-Band
            TB_input_vector[:,14:15], # W-Band
            TB_input_vector[:,15:19], # F-Band
            TB_input_vector[:,19:]), # G-Band
            axis=1)
    TB_input_vector.shape

    if (len(t_steps)*len(cell_selection)!=(TB_input_vector.shape[0])):
        print("Shape of TB does not match cell and time steps. Please check.")

        """
        # exclude profiles with unrealistic pamtra simulations
        TBs[TBs[:,20]<230] = np.nan   # TODO what are in our case unrealistic values?
        TB_input_vector = TBs[~np.isnan(TBs).any(axis=1),:] #TODO: count NANs. If NAN do not just drop , as index would get messed up
        IWP = IWP[~np.isnan(TBs).any(axis=1)]
        LWP = LWP[~np.isnan(TBs).any(axis=1)]
        IWV = IWV[~np.isnan(TBs).any(axis=1)]
        """
        if np.count_nonzero(np.isnan(TBs)) > 0:
            print("NaN values in Brightnesstemperature array. This may influence the rest of the Retrievaldeveloment, as they are not being filtered.")
    train_TB, test_TB, validate_TB = splitting_in_train_test_validate(TB_input_vector,t_steps,slices_train,slices_validate,slices_test)
    train_TB=np.concatenate([train_TB, validate_TB])

    TBs_train_scaled, mu_train, sigma_train = standardize_nn_training_data(train_TB) 
    TBs_test_scaled = standardize_nn_input_data_v2(test_TB,mu=mu_train,sigma=sigma_train) 
    TBs_validate_scaled = standardize_nn_input_data_v2(validate_TB,mu=mu_train,sigma=sigma_train)

    print("altitude used for training:", altitude)

    # compile network
    dnn_model_hyd = tf.keras.Sequential([
        tf.keras.layers.Dropout(rate=0.05, input_shape=(train_TB.shape[1],)),
        tf.keras.layers.Dense(NR_of_NEURONS_L1,
                              input_shape=(train_TB.shape[1],),
                              bias_initializer=BIAS_INIT,
                              kernel_initializer=WEIGHT_INIT,
                              kernel_regularizer="l2",
                              activation=ACTIVATION_HL,
                             ),
        tf.keras.layers.Dense(1,
                              activation=ACTIVATION_OP)])
    
    dnn_model_hyd.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                          loss=LOSS_FUNCTION,
                         )
    
    dnn_model_hyd.summary()

    
    
    #train network
    history = dnn_model_hyd.fit(
        TBs_train_scaled, np.sqrt(train_hyd),
        validation_split=0.2,
        #validation_data=(TBs_validate_scaled,validate_hyd),#new
        verbose=0,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)]
    )
    
    # use typhon ploting style
    #plt.style.use(ty.plots.styles.get('typhon'))
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim(0,40)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    plt.title(variable)

    # Visualization of predictions
    hyd_test_predictions = dnn_model_hyd.predict(TBs_test_scaled)[:,0]
    hyd_test_predictions_squared =hyd_test_predictions**2
    fig, ax = plt.subplots()

    labels = [
        f"Truth",
        "Prediction",
        "Prediction (cliped)"]
    ax.hist(np.array([np.sqrt(test_hyd),hyd_test_predictions]).T,bins=25,label=labels)#,IWP_test_predictions_cliped
    ax.set_xlabel("$\sqrt{IWP}$")
    ax.set_ylabel("Frequency")
    plt.legend()
    plt.show()

    dnn_model_hyd.save(f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/NNs_{variable}/dnn_model_{variable}_24-32-1_reg_{altitude}m_{training_version}.keras')
    #save mu and sigma
    np.save(f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/standardizing_parameters/mu_' + name_pamtra_run + training_version + '_'+ str(altitude) +'m.npy',mu_train)
    np.save(f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/standardizing_parameters/sigma_' + name_pamtra_run + training_version + '_'+ str(altitude) +'m.npy',sigma_train)
    print("NN for",altitude,"m has been saved.")
    if altitude == 14450:
        train_hyd=np.save(f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/NNs_{variable}/dnn_model_{variable}_24-32-1_reg_{altitude}m_ + {training_version}_train_{variable}.npy',train_TB)
        test_hyd=np.save(f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/NNs_{variable}/dnn_model_{variable}_24-32-1_reg_{altitude}m_ + {training_version}_test_{variable}.npy',test_TB)
        train_TB=np.save(f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/NNs_{variable}/dnn_model_{variable}_24-32-1_reg_{altitude}m_ + {training_version}_train_TB.npy',train_hyd)  #Hyd to train
        test_TB=np.save(f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/NNs_{variable}/dnn_model_{variable}_24-32-1_reg_{altitude}m_ + {training_version}_test_TB.npy',test_hyd)
        print("Training and Test data for",altitude,"m has been saved.")

#%% Retrieval Application


ds_HAMP=xr.open_dataset("/work/um0203/u301032/master_thesis/flight_data/halo_HAMP.nc")
ds_iwv_kw=xr.open_dataset("/work/um0203/u301032/master_thesis/flight_data/halo_iwv_kw.nc")
ds_sondes=xr.open_dataset("/work/um0203/u301032/master_thesis/flight_data/halo_sondes.nc")
ds_halo_altitude=xr.open_dataset("/work/um0203/u301032/master_thesis/flight_data/halo_altitude.nc") 

variable='IWV'
print(ds_HAMP.frequency)
ds_hamp=ds_HAMP

excluded_frequencies = ds_hamp.frequency.where(ds_hamp.frequency!=184.81, drop=True)

# Step 2: Filter the dataset to exclude the specified frequency
ds_hamp = ds_hamp.sel(frequency=excluded_frequencies)
print(ds_hamp.TBs)


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def retrieve_hyd(TBs,altitudes):
    
    levels = np.array([find_nearest_value(flight_levels,altitudes[i]) for i in range(len(altitudes))])
    hyd== np.zeros(len(TBs))
    hyd[:] = np.nan
    for i in range(len(flight_levels)):
        altitude =flight_levels[i]
        if len(levels[levels==altitude]) != 0:
            print(f"Retrieving from {altitude} ({len(levels[levels==altitude])} TBs)")
            #hier modul load
            mu_train=np.load(f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/standardizing_parameters/mu_' + name_pamtra_run + training_version + '_'+ str(altitude) +'m.npy')
            sigma_train=np.load(f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/standardizing_parameters/sigma_' + name_pamtra_run + training_version + '_'+ str(altitude) +'m.npy')
            
            ds_hamp_scaled=standardize_nn_input_data_v2(np.asarray(ds_hamp.TBs[levels==altitude].values),mu=mu_train,sigma=sigma_train)

            dnn_model_hyd = tf.keras.models.load_model(f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/NNs_{variable}/dnn_model_{variable}_24-32-1_reg_{altitude}m_ + {training_version}.keras',compile=False) #TODO adjust input name of retrieval
            hyd[np.where(levels==altitude)] = dnn_model_hyd.predict(ds_hamp_scaled)[:,0]**2
    # Indicate the fraction of negative predictions (possible as the sqrt of hyd is retrieved).
    print("")
    print((len(hyd[hyd<0])/len(hyd)),"% negative predictions (=clipped to 0).")
    print("")
    # If negative  values exist, they are clipped to 0.
    hyd[hyd<0] = 0.
    # clip everything below 10500km
    hyd[altitudes<=10500] =np.nan
    levels[altitudes<=10500] =0

    return hyd, levels 
IWV, levels =retrieve_hyd(ds_hamp['TBs'].values,ds_hamp['plane_altitude'].values)