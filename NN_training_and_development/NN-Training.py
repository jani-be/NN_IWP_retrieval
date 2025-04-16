# Development of Retrieval
# Environment to use: Python3-unstable

'''
GOAL: Retrieval of 1. IWP 2. IWV 3. LWP (Maybe) 4. RWP (even less maybe)
Training and Testing of Neural Network

# Reading Data
# Splitting in Test (, Validation) and Training data
# Construct NN -> What is best architecture? Guess: 15-32 hidden Neurons
# How to determine what architecture is best?


LOAD AND PREPROCESS TRAINING DATA
# CREATE NEW SET OF  TRAININGS DATA

DEFINITION AND COMPILATION OF NETWORK
# TRAINING OF NEURAL NETWORK
## plot of loss and val_loss

NN-IWP PERFORMANCE ON PAMTRA TEST DATASET
# PREDICT AND CLIP DATA
# ANALYSIS:
## Truth vs Prediction Histogram
## Truth vs Prediction Scatter with Bias, RMSE and Corr
## Truth vs Prediction Scatter with distribution
## Truth vs Prediction distribution

SAVING NN-IWP MODEL/RETRIEVAL TO DISK 

IDEAS FOR LATER ON:
# Later on Test with Halo.
# Cloud Mask as an advantage?
'''


###


#%% Modules to load

#import tensorflow as tf
##import levenberg_marquardt as lm
#import scipy.stats
#from sklearn.preprocessing import MinMaxScaler
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import seaborn as sns
from glob import glob
import xarray as xr
import pandas as pd
from functools import partial
##import typhon as ty
import sys
sys.path.append('/home/u/u301238/master_thesis/')
sys.path.append('/home/u/u301032/orcestra/NN_IWP_retrieval/NN_training_and_development/')
sys.path.append('/home/u/u301032/orcestra/NN_IWP_retrieval/')

#import src
import src_comparison_halo_pamtra as chp
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# use typhon ploting style
#plt.style.use(ty.plots.styles.get('typhon'))


#%%
dates=["0824","0829","0927"]
appendices=["-rerun","-high3Drate","-rerun"]
name_pamtra_run = "all_area_1000th_cell" # "cells_025x025_2h"
if name_pamtra_run == "all_area_1000th_cell":
    # Pamtra files
    time_selection = "4h"
    cell_selection = np.load('/home/u/u301032/orcestra/NN_IWP_retrieval/NN_training_and_development/cells_all_area_1000th_cell.npy')
    pamtra_files ='/work/um0203/u301032/PAMTRA_output/PAMTRA-ICON_*_all_area_v1.nc'
elif name_pamtra_run == "cells_025x025_2h":
    cell_selection = np.load('/home/u/u301032/orcestra/NN_IWP_retrieval/NN_training_and_development/cells_025x025_sea.npy')
    time_selection = "2h"
    #pamtra_files = 
else:
    print('please insert correct name of pamtra run')
# Corresponding icon 2D files


altitude = 13900 #ORCESTRA  12000 #AC3 #Example heights #TODO: Automate for all flightlevels 
# CAUTION !! CHECK IF PAMTRA FILES MATCH SELECTION CRITERIA !!
#%% LOAD AND PREPROCESS TRAINING DATA
# CREATE NEW SET OF  TRAININGS DATA
#%%




#Read in all 2D fields.


# Getting list of 2D files to use
path_sim = "/work/mh0492/m301067/orcestra/icon-mpim/build-lamorcestra/experiments/"
twodim_files=[]
for DATE, appendix in zip(dates,appendices):
    twodim_files.append(path_sim + f"orcestra_1250m_{DATE+appendix}/orcestra_1250m_{DATE+appendix}_atm_2d_ml_DOM01_2024{DATE}T000000Z.nc")

#Reading and First Processing of datasets
partial_func = partial(_preprocess,cells=cell_selection,frequency=time_selection) 
ds_icon_2d= xr.open_mfdataset(twodim_files,preprocess=partial_func)#, chunks={"ncells": -1})#,chunks="auto", parallel=True)




# load training data Max
#TBs, frozen_water, liquid_water, IWV = load_nn_training_data(pamtra_files,altitude=altitude) #TODO Adjust Code to my needs - load testing data aswell?
# me
TBs, IWV, IWP, LWP = chp.load_nn_training_data(pamtra_files,dates,appendices,altitude=13900) #TODO Adjust Code to my needs - load testing data aswell?

#%%

# exclude profiles closer than 5km to ICON-HALO collocated profiles
#collocation_profiles_mask = np.load('/home/u/u301238/master_thesis/nn/collocation_profiles_mask.npy')  #TODO Kick out 
#TBs = TBs[collocation_profiles_mask==0]
#IWP = IWP[collocation_profiles_mask==0]
#LWP = LWP[collocation_profiles_mask==0]
#IWV = IWV[collocation_profiles_mask==0]

# exclude profiles with unrealistic pamtra simulations
TBs[TBs[:,20]<230] = np.nan   # TODO what are in our case unrealistic values?
TB_input_vector = TBs[~np.isnan(TBs).any(axis=1),:] #TODO: count NANs. If NAN do not just drop , as index would get messed up
IWP = IWP[~np.isnan(TBs).any(axis=1)]
LWP = LWP[~np.isnan(TBs).any(axis=1)]
IWV = IWV[~np.isnan(TBs).any(axis=1)]

# set IWP values below 1gm2 to zero #TODO check what this means in tropics eg literature, histograms
IWP[IWP<1.]=0.

# concatenate TB vector manually #TODO how changes this the vector? why hasn't been done before. Only is done for TB, not for hydloads
TB_input_vector = np.concatenate((
        TB_input_vector[:,0:7], # K-Band
        TB_input_vector[:,7:14], # V-Band
        TB_input_vector[:,14:15], # W-Band
        TB_input_vector[:,15:19], # F-Band
        TB_input_vector[:,19:]), # G-Band
        axis=1)
    
# split training data into train and test subsets
#TBs_train, TBs_test, IWP_train, IWP_test  = src.split_nn_training_data(TB_input_vector,IWP,split_ratio=0.75)
TBs_train, TBs_test, IWP_train, IWP_test, LWP_train, LWP_test, IWV_train, IWV_test = src.split_nn_training_data(TB_input_vector,IWP,LWP=LWP,IWV=IWV,split_ratio=0.75)
# standardize all TBs along their respective channel
# as a normal distribution with mean 0 and std of 1
TBs_train_scaled, mu_train, sigma_train = src.standardize_nn_training_data(TBs_train) #TODO write summary of function. Why is it needed? wht does it do? TODO Adapt function
TBs_test_scaled = src.standardize_nn_input_data_v2(TBs_test,mu=mu_train,sigma=sigma_train) #TODO write summary of function. How does ist interact with the previous one?  TODO Adapt function


#%% DEFINITION AND COMPILATION OF NETWORK

# General settings of the neural network #TODO Later on: Outline Training Strategy: what has been done, what is promising? How to check results?
NR_of_NEURONS_L1 = 32
BIAS_INIT = 'zeros'
WEIGHT_INIT = 'random_normal'
ACTIVATION_HL = 'tanh'
ACTIVATION_OP = 'linear'
LOSS_FUNCTION = tf.keras.losses.MeanSquaredError()
LEARNING_RATE = 0.001

EPOCHS = 1000
BATCH_SIZE = 50

# define & compile nn-model for iwp with specified settings
# * 1 input layer (24 neurons) 
# * 1 hidden layer (64 neurons)
# * 1 output layer (1 neuron)

dnn_model_iwp = tf.keras.Sequential([
    tf.keras.layers.Dropout(rate=0.05, input_shape=(TBs_train.shape[1],)),
    tf.keras.layers.Dense(NR_of_NEURONS_L1,
                          input_shape=(TBs_train.shape[1],),
                          bias_initializer=BIAS_INIT,
                          kernel_initializer=WEIGHT_INIT,
                          kernel_regularizer="l2",
                          activation=ACTIVATION_HL,
                         ),
    tf.keras.layers.Dense(1,
                          activation=ACTIVATION_OP)])

dnn_model_iwp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=LOSS_FUNCTION,
                     )

dnn_model_iwp.summary()

#%% TRAINING OF NEURAL NETWORK


history = dnn_model_iwp.fit(
    TBs_train_scaled, np.sqrt(IWP_train),
    validation_split=0.2,
    verbose=0,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)]
)

# use typhon ploting style
#plt.style.use(ty.plots.styles.get('typhon'))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim(0,40)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid(True)
plt.legend()

#%% NN-IWP PERFORMANCE ON PAMTRA TEST DATASET
# PREDICT AND CLIP DATA
# ANALYSIS: #TODO: if same style is wished, divide script in two parts
## Truth vs Prediction Histogram
## Truth vs Prediction Scatter with Bias, RMSE and Corr
## Truth vs Prediction Scatter with distribution
## Truth vs Prediction distribution

#%% SAVING NN-IWP MODEL/RETRIEVAL TO DISK 
#TODO: how to save, looks complicated!