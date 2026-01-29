# These fcts are used for NN_development.ipynb
# uses ML as environment

import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

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
from matplotlib import cm
from scipy.interpolate import interpn
from matplotlib.colors import Normalize 

import sys
#sys.path.append('/home/u/u301238/master_thesis/')
sys.path.append('/home/u/u301032/orcestra/NN_IWP_retrieval/NN_training_and_development/')
print(sys.path)

def array_setup(TBs,t_steps,cell_selection):
    TB_input_vector=TBs
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


    return TB_input_vector


def plot_NN_error_v4(true, prediction,variable,ax=None,filename=None):
    if variable=='CLWP':
        variable='LWP' #for eays plotting
    #so far used for graphics
    a,b=dev.no_nan_for_plot(true, prediction)
    # Define bin edges (e.g., 10 bins)
    num_bins = 20
    
    if variable == 'IWV':
        num_bins = 22
        bins = np.linspace(17.25, 72.25, num_bins+1 )#np.linspace(np.min(b), np.max(b), num_bins+1 )
    else: 
        if variable == 'IWP':
            bin_max = 4.25 # max(a) #set to number of interest or maybe with 
        if variable == 'LWP':
            bin_max = 4.25
        
        bin_min = 0
        num_bins = int((bin_max - bin_min) *4)
        bins  = np.logspace(bin_min, bin_max, num_bins + 1)
    # 
    
    # Digitize 'a' to find out which bin each value belongs to
    bin_indices = np.digitize(b, bins)
    
    # Initialize arrays to hold results
    bin_centers = (bins[:-1] + bins[1:]) / 2
    b_error = np.empty(num_bins)
    b_bias = np.empty(num_bins)
    b_std = np.empty(num_bins)
    
    rel_100 = np.empty(num_bins)
    # Calculate error (e.g., standard deviation) of 'b' within each bin
    for i in range(1, num_bins + 1):
        
        # Find indices of data points in the current bin
        in_bin = bin_indices == i

        if np.sum(in_bin) <=5:
            b_error[i-1]=np.nan
            b_bias[i-1]=np.nan
            print(np.log10(bin_centers[i-1]))
            continue
        #error as rmse between true and predicted # like in marek s paper
        targets= a[in_bin]
        predictions= b[in_bin]
        b_error[i-1]= np.sqrt(((predictions - targets)**2).mean())
        #rel_100[i-1]= np.sqrt((((predictions - targets)/predictions)**2).mean())
        b_bias[i-1]= (predictions - targets).mean()
        ## Calculate standard deviation of 'b' for these points
        #b_std[i - 1] = np.std(predictions - b_bias[i-1])
        #rel_100=

        print(np.sum(in_bin),'bias: ',b_bias[i-1],'RMSE: ',b_error[i-1],np.log10(bin_centers[i-1]))
    
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,6))  
        show_plot = True
    else:
        show_plot = False
    # Plot Error curves
    
    
    if variable == 'IWV':
        error_curves =[1,5,10]
        x=np.linspace(np.min(a), np.max(a),200)
    else:
        error_curves =[5,10,25,50,100]
        x=np.logspace(bin_min, bin_max, num_bins *10)
    for e in error_curves:
        ax.plot(x,-(0.01*e)*x,linestyle=':',color='grey',label=e)
        ax.plot(x,(0.01*e)*x,linestyle=':',color='grey',label=e) #relative erorr 20 percent
        if variable=='IWV':
            X=np.min(b)+1
            Y=X*((0.01*e))
            print(e,Y,X)
            #plt.scatter(X, Y)
            ax.text(X, Y,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=0)
        else:
            X=950#1200#400 + (0.01*e)*800
            ax.text(X*(1/(0.01*e)), X,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=80)
            #plt.scatter(X*(1/(0.01*e)), X)
    ax.plot(x,0*x,linestyle='-',color='grey',label=e)

        
    ax.plot(bin_centers,b_error)
    ax.plot(bin_centers,b_bias)
    #plt.errorbar(bin_centers, np.zeros_like(bin_centers), yerr=b_error, fmt='o', capsize=5)

    if variable == 'IWV':
        ax.set_xlabel('Retrieved '+ variable+' \\ kg m$^{-2}$')
        ax.set_ylabel('Error: Retrieved - true '+variable+' \\ kg m$^{-2}$')
        ax.set_ylim(bottom=-1,top=7)
        
        #plt.plot(bin_centers,0.05*bin_centers,linestyle='--',color='grey') #relative erorr 100 percent
        #plt.plot(bin_centers,0.01*bin_centers,linestyle='-.',color='grey') #relative erorr 100 percent
    else:
        ax.set_ylabel('Error: Retrieved - true '+variable+' \\ g m$^{-2}$')
        ax.set_xscale('log')
        #plt.yscale('log')
        ax.set_xlabel('Retrieved '+ variable+' \\ g m$^{-2}$')
        
        if  variable == 'IWP':
            ax.set_ylim(bottom=-350,top=1050) #(bottom=-320,top=1020) #
        elif  variable == 'LWP':
            ax.set_ylim(bottom=-350,top=1050) #(bottom=-220,top=1000) 
            #plt.yscale('log')
        else:
            ax.set_ylim(top=120)       

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #plt.title('Error of b within each a bin')
    
    if filename and show_plot:
        plt.tight_layout()
        plt.savefig(f'/home/u/u301032/orcestra/plots/{variable}_error_biasRMSE_{filename}.png',dpi=400)
    if show_plot:
        plt.tight_layout()
        plt.show()


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

def defining_slices_for_data():
    """
    delivers arraywith integer for splitting function 
    """
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
    for i in range(len(slices_validate)):
        print()
        #print(slices_train[2*i])
        #print(slices_validate[i])
        #print(slices_train[2*i+1])
        #print(slices_test[i])

    return slices_train,slices_validate,slices_test

def defining_slices_for_data_v2():
    """
    delivers arraywith integer for splitting function 
    """
    a=np.arange(0,33,3)
    l=[]
    for i in range(len(a)):
        l.append([a[i],a[i]+1])
    slices_validate=np.asarray(l)    
    a=np.arange(1,33,3)
    l=[]
    for i in range(len(a)):
        l.append([a[i],a[i]+1])
    slices_test=np.asarray(l)   
    a=np.arange(2,33,3)
    l=[]
    for i in range(len(a)):
        l.append([a[i],a[i]+1])
    slices_train=np.asarray(l)    
    for i in range(len(slices_validate)):
        print()
        #print(slices_validate[i])
        #print(slices_test[i])
        #print(slices_train[i])
        
        #print(slices_train[2*i+1])
        

    return slices_train,slices_validate,slices_test

def splitting_train_test_2(hyd,t_steps):
    slices_train, slices_validate, slices_test = defining_slices_for_data_v2()
    hyd1, hyd2, hyd3 = splitting_in_train_test_validate(hyd,t_steps,slices_train,slices_validate,slices_test)
    train_hyd=np.concatenate([hyd1, hyd3])
    test_hyd=hyd2
    return train_hyd, test_hyd


def plot_3_by_1(data, plottype, label, variables = ['IWV','LWP','IWP'],savefig=False, filename = None):
    """
    convenient plotting for IWV LWP and IWP
    plot types implemented so far:
    - hist
    data : if type hist, then format in the shape of [[train_IWV, test_IWV],[train_LWP, test_LWP],[train_IWP, test_IWP]]
    """
    plt.rcParams.update({'font.size': 14}) # labels in 16
    plt.rcParams['savefig.dpi'] = 400
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loop over data and corresponding axes:
    for ax, d, variable, title in zip(axes, data, variables, ['a)','b)','c)']):
        if plottype == 'hist':
            ax.hist(d,label=label,density = True,log=True)
            ax.set_xlabel(variable, fontsize=16)
        ax.legend(label)    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
                
        ax.text(0.05, 1.03, title, transform=ax.transAxes,
          fontsize=16,  va='top')
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=12))
        
    # Improve layout and display
    plt.tight_layout()
    filename= "hist_traing_testing_set"
    if savefig:
        if filename is not None:
            plt.savefig(f'/home/u/u301032/orcestra/plots/{filename}.png')
            print('Plot saved')
        else:
            print('Please define filename. Otherwise saving of plot is not possible')        
    plt.show()

def plot_3_by_1_errors(target_data, prediction_data, variables=['IWV','LWP','IWP'], filename=None):

    # Erstellen der Figure mit 3 Unterplots in einer Spalte
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))#15,5
    
    # Für jedes Variable: plot auf dem jeweiligen Subplot
    for ax, var, title in zip(axes, variables, ['a)','b)','c)']):
        # Aufrufen der angepassten Funktion

        plot_NN_error_v4(target_data[var], prediction_data[var], var, filename=filename, ax=ax)
        # Optional: Titel oder andere Anpassungen
        
        ax.text(0.05, 1.03, title, transform=ax.transAxes,
          fontsize=16,  va='top')
    
    plt.tight_layout()
    if filename:
        plt.savefig(f'/home/u/u301032/orcestra/plots/RMSEandbias_3x1{filename}.png')
        print('Plot gespeichert als:', f'/home/u/u301032/orcestra/plots/RMSEandbias_3x1{filename}.png')
    plt.show()

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
        

def define_and_compile_nn(TBs_train, NR_of_NEURONS_L1, BIAS_INIT, WEIGHT_INIT, ACTIVATION_HL, ACTIVATION_OP, LOSS_FUNCTION, LEARNING_RATE):
    """
    # define & compile nn-model for lwp with specified settings
    # * 1 input layer (24 neurons) 
    # * 1 hidden layer (64 neurons)
    # * 1 output layer (1 neuron)
    """    
    dnn_model = tf.keras.Sequential([
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
    
    dnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                          loss=LOSS_FUNCTION,
                         )
    
    dnn_model.summary()
    return dnn_model

def train_NN(dnn_model, TBs_train_scaled, train_hyd, EPOCHS, BATCH_SIZE, plotting = False):
    history = dnn_model.fit(
        TBs_train_scaled, np.sqrt(train_hyd),
        validation_split=0.2,
        #validation_data=(TBs_validate_scaled,validate_hyd),#new
        verbose=0,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)]
    )
    if plotting:
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        #plt.ylim(0,40)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
        
        #if saving:
        #plt.savefig('/home/u/u301238/master_thesis/plots/NN/NN_IWP_training_error_loss_24-32-1.jpg',bbox_inches='tight',dpi=200)
    return dnn_model

def clip_nn_output(nn_prediction,truth=None):
    
    negative_predictions = np.round((len(nn_prediction[nn_prediction<0.])/len(nn_prediction))*100,2)
    print(f"Negative predictions: {negative_predictions} %")

    # clip nn output / predictions (= set negative values to zero)
    nn_prediction_cliped = nn_prediction.copy()
    nn_prediction_cliped[nn_prediction_cliped<0.]=0.
    
    if truth is not None:
        # calculate bias before and after cliping
        bias_before = np.mean(nn_prediction) - np.mean(truth)
        bias_after = np.mean(nn_prediction_cliped) - np.mean(truth)

        print("Bias before cliping: ",np.round(bias_before,2))
        print("Bias after cliping: ",np.round(bias_after,2))

    return nn_prediction_cliped


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def NN_train_save(TBs_train_scaled,TBs_test_scaled, train_hyd,test_hyd,variable,altitude,training_version,parameters):
    EPOCHS, BATCH_SIZE, NR_of_NEURONS_L1, BIAS_INIT, WEIGHT_INIT, ACTIVATION_HL, ACTIVATION_OP, LOSS_FUNCTION, LEARNING_RATE =parameters
    # define andcompile model
    dnn_model = define_and_compile_nn(TBs_train_scaled, NR_of_NEURONS_L1, BIAS_INIT, WEIGHT_INIT, ACTIVATION_HL, ACTIVATION_OP, LOSS_FUNCTION, LEARNING_RATE)
    # shuffle NN input for validation with randomized data
    TBs_train_scaled_shuffle, train_hyd_shuffle=unison_shuffled_copies( TBs_train_scaled, train_hyd)
    # train model
    dnn_model = train_NN(dnn_model, TBs_train_scaled_shuffle, train_hyd_shuffle, EPOCHS, BATCH_SIZE, plotting = False)
    #save model
    path=f'/home/u/u301032/orcestra/NN_IWP_retrieval/NNs/NNs_{variable}'
    dnn_model.save(path+f'/dnn_model_{variable}_24-32-1_reg_{altitude}m_{training_version}.keras')
    print("NN for",altitude,"m has been saved.")
    
    if altitude == 14450:
        np.save(path+f'/dnn_model_{variable}_24-32-1_reg_{altitude}m_{training_version}_train_{variable}.npy',train_hyd)
        np.save(path+f'/dnn_model_{variable}_24-32-1_reg_{altitude}m_{training_version}_test_{variable}.npy',test_hyd)
        np.save(path+f'/dnn_model_{variable}_24-32-1_reg_{altitude}m_{training_version}_train_TB_scaled.npy',TBs_train_scaled)  #Hyd to train
        np.save(path+f'/dnn_model_{variable}_24-32-1_reg_{altitude}m_{training_version}_test_TB_scaled.npy',TBs_test_scaled)
        print("Training and Test data for",altitude,"m has been saved.")


def plot_truth_prediction_sqrt(test_hyd, test_predictions, test_predictions_cliped, variable):
    fig, ax = plt.subplots()
    
    labels = [
        f"Truth",
        "Prediction",
        "Prediction (cliped)"]
    ax.hist(np.array([np.sqrt(test_hyd), test_predictions, test_predictions_cliped]).T,bins=40,label=labels)#,IWP_test_predictions_cliped
    #ax.text(35.,3000,"$\overline{Truth}=$"+str(np.round(np.nanmean(np.sqrt(IWP_test)),2))+"$\sqrt{g}m^{-1}$",ha="left")
    #ax.text(35.,2700,"$\overline{Prediction}=$"+str(np.round(np.mean(IWP_test_predictions),2))+"$\sqrt{g}m^{-1}$",ha="left")
    ##ax.text(35.,2400,"$\overline{Prediction (cliped)}=$"+str(np.round(np.mean(IWP_test_predictions_cliped),2))+"$\sqrt{g}m^{-1}$",ha="left")
    if variable == 'LWP':
        ax.set_xlabel(r"$\sqrt{LWP}$")
    if variable == 'IWP':
        ax.set_xlabel(r"$\sqrt{IWP}$")
    if variable == 'IWV':
        ax.set_xlabel(r"$\sqrt{IWV}$")
    ax.set_ylabel("Frequency")
    plt.legend()
    plt.show()
    #plt.savefig('/home/u/u301238/master_thesis/plots/NN/NN_IWP_predictions_cliping_effect.jpg',bbox_inches='tight',dpi=200)



def crossvalidation(TBs_data,hyd_data):
    # no improvement for IWP
    # Parameters
    num_folds = 5
    kfold = KFold(n_splits=num_folds)
    
    # Store evaluation scores here
    val_scores = []
    # Build new model per fold
    dnn_model = dev.define_and_compile_nn(
        TBs_data, 
        NR_of_NEURONS_L1, BIAS_INIT, WEIGHT_INIT, 
        ACTIVATION_HL, ACTIVATION_OP, LOSS_FUNCTION, LEARNING_RATE
    )
    for fold, (train_idx, val_idx) in enumerate(kfold.split(TBs_data)):
        print(f'---- Fold {fold+1} of {num_folds} ----')
    
        # Train/Val Split
        TBs_train, hyd_train = TBs_data[train_idx], hyd_data[train_idx]
        TBs_val, hyd_val     = TBs_data[val_idx], hyd_data[val_idx]
        print(TBs_train)
        # Standardize (fit ONLY on train, apply to train and val!)
        TBs_train_scaled, mu_train, sigma_train = dev.standardize_nn_training_data(TBs_train)
        TBs_val_scaled = dev.standardize_nn_input_data_v2(TBs_val, mu=mu_train, sigma=sigma_train)
    
        
    
        # Train model (No further split to validation, since val comes from k-fold)
        history = dnn_model.fit(
            TBs_train_scaled, np.sqrt(hyd_train),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)]
        )
    
        # Evaluate on validation fold
        y_pred = dnn_model.predict(TBs_val_scaled)#.flatten()
        y_true = np.sqrt(hyd_val)
        fold_mse = np.mean((y_pred - y_true) ** 2)
        print(f'Fold {fold+1} MSE: {fold_mse:.4f}')
        val_scores.append(fold_mse)
        
        plt.plot(history.history['loss'], label='loss')
        plt.show()
        
        #plt.ylim(0,40)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
    
    print('===== Cross-validation Results =====')
    print(f'Mean MSE: {np.mean(val_scores):.4f}, Std: {np.std(val_scores):.4f}')
    return dnn_model
    

def density_scatter( x , y, ax = None, sort = True, bins = 20,title="title",xlabel="x",ylabel="y",lim=None, **kwargs )   :
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
    if lim != None:
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    cbar.ax.set_ylabel('Density')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(str(title))

    return ax

def scatter_hyd(a,b,variable):
    nans = np.logical_or(np.isnan(a), np.isnan(b))
    a = a[~nans]
    b = b[~nans]
    bias = np.round((np.mean(b) - np.mean(a)),2)
    corr = np.round((scipy.stats.pearsonr(a,b)[0]),2)
    rmse = np.round(np.sqrt(np.nanmean((b-a)**2)),2)
    print('bias: ',bias,' corr: ',corr,' rmse: ',rmse)
    if variable == 'IWV':
        density_scatter(a,b,xlabel=f'True {variable}'+' [kg m$^{-2}$]',ylabel=f'Retrieved {variable}'+' [kg m$^{-2}$]',title=variable )
    
    else:
        density_scatter(a,b,xlabel=f'True {variable}'+' [g m$^{-2}$]',ylabel=f'Retrieved {variable}'+' [g m$^{-2}$]',title=variable )
    
    mi=np.min([a,b])
    ma=np.max([a,b])
    d=np.abs(mi-ma)*0.05
    plt.xlim([mi-d,ma+d])
    plt.ylim([mi-d,ma+d])
    plt.show()

def plot_scatter_3_tiles(filename=None):
    fig = plt.figure(figsize=(15, 15),constrained_layout=True)

    subfigs = fig.subfigures(2, 2)
    subfigs[0,1]

    for var, subfig, title in zip(variables,subfigs.flat, ['a)','','b)','c)']):
        if subfig ==subfigs[0,1]:
            ax=subfig.subplots()
            ax.axis('off')
        else:
            if var =='IWV':
                kind='lin'
            else: kind='log'    
            plot_scatter_log( target_data[var],prediction_data[var], var,kind = kind,fig=subfig)
            subfig.suptitle(' ',ha='left')
            subfig.text(0.0, 1.0, title, #transform=ax.transAxes,
                                       fontsize=16,  va='top')
    if filename:
        #plt.tight_layout()
        plt.savefig(f'/home/u/u301032/orcestra/plots/scatter_tiles_target_predictions_{filename}.png',dpi=400)
    plt.show()

def plot_scatter_log( test_hyd,hyd_test_predictions_squared, variable,kind = 'log',plotname=None,fig=None):
    if variable=='CLWP':
        variable ='LWP' #for plotting correct name
    if fig == None:
        #fig, axs = plt.subplots(figsize=(9,8))
        fig, axs = plt.subplot_mosaic([['histx', '.'],
                                ['scatter', 'histy']],
                                figsize=(6, 6),
                                width_ratios=(5, 1), height_ratios=(1, 5),
                                layout='constrained')
    else:
        axs = fig.subplot_mosaic([['histx', '.'],
                                ['scatter', 'histy']],
                                 width_ratios=(5, 1), height_ratios=(1, 5))


        
    if kind == 'log':
        
        test_hyd_log = test_hyd.copy()
        test_hyd_log[test_hyd_log==0] = 10**(-16)
        hyd_test_predictions_squared_log = hyd_test_predictions_squared.copy()
        hyd_test_predictions_squared_log[hyd_test_predictions_squared_log==0] = 10**(-16)
        
        a = np.log10(test_hyd_log)
        b = np.log10(hyd_test_predictions_squared_log)
        if variable != 'IWV':
            a[a<0.] = 0
            b[b<0.] = 0
        
        #a[a<0.] = np.nan
        #b[b<0.] = np.nan
        
        #a = test_hyd.copy()
        #b = hyd_test_predictions_squared.copy()
        
        #a[a<1.] = 1.
        #b[b<1.] = 1.
    if kind=='lin':
        a = (test_hyd)
        b = (hyd_test_predictions_squared)
        a[a<0.] = 0
        b[b<0.] = 0
    
    nans = np.logical_or(np.isnan(a), np.isnan(b))
    a = a[~nans]
    b = b[~nans]
    
    maxval = max(np.max(a),np.max(b))
    minval = max(np.min(a),np.min(b))
    
    if variable == 'IWV':
        minval-=5
        maxval+=5
    else:
        minval_lin=0.7
        maxval_lin=10**(maxval+0.15)
        
    # Calculate the point density
    ab = np.vstack([a,b])
    c = scipy.stats.gaussian_kde(ab)(ab)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = c.argsort()
    a, b, c = a[idx], b[idx], c[idx]
    
    bias = np.round((np.mean(b) - np.mean(a)),2)
    corr = np.round((scipy.stats.pearsonr(a,b)[0]),2)
    rmse = np.round(np.sqrt(np.nanmean((b-a)**2)),2)
    
    axs['scatter'].grid(alpha=0.5,which='both')
    if variable != 'IWV':
        hyd_true = np.arange(1,np.nanmax(test_hyd),0.01)
    else:
        hyd_true = np.arange(minval,maxval,0.01)
    RE_0 = hyd_true
    RE_80_plus = (0.8*hyd_true)+hyd_true
    RE_80_minus = (-0.8*hyd_true)+hyd_true
    RE_50_plus = (0.5*hyd_true)+hyd_true
    RE_5_plus = (0.05*hyd_true)+hyd_true
    RE_5_minus = (-0.05*hyd_true)+hyd_true
    RE_50_minus = (-0.5*hyd_true)+hyd_true
    RE_20_plus = (0.2*hyd_true)+hyd_true
    RE_20_minus = (-0.2*hyd_true)+hyd_true
    RE_20_minus[RE_20_minus<1.]=np.nan
    RE_50_minus[RE_50_minus<1.]=np.nan
    RE_80_minus[RE_80_minus<1.]=np.nan
      
    #axs.scatter(y_test, y_test_predictions**2)
    # no labels

        
    if kind == 'log':
        #sc = axs['scatter'].scatter(10**a,10**b,c=c,s=15,cmap='viridis',vmin=0,vmax=0.3)
        sc = axs['scatter'].scatter(10**a,10**b,c=c,s=15,cmap='viridis',vmin=0,vmax=0.3)
        #sc = axs.scatter(a,b,c=c,s=8,cmap='viridis',vmin=np.min(c),vmax=np.max(c))
        
        
        #axs.plot(np.arange(0,maxval),np.arange(0,maxval),linewidth=3,color='black',alpha=1,label='1:1')
        axs['scatter'].set_ylim(bottom=minval_lin,top=maxval_lin)
        axs['scatter'].set_xlim(left=minval_lin,right=maxval_lin)

        axs['histy'].set_ylim(bottom=minval_lin,top=maxval_lin)
        axs['histx'].set_xlim(left=minval_lin,right=maxval_lin)
        
        axs['scatter'].set_xscale('log')
        axs['scatter'].set_yscale('log')
        axs['scatter'].set_xlabel(f'Target {variable} \\ g m$^{-2}$')
        axs['scatter'].set_ylabel(f'NN Predicted {variable} \\ g m$^{-2}$')
        if variable == 'IWV':
            axs['scatter'].plot(np.arange(10**minval,10**maxval),np.arange(10**minval,10**maxval),linewidth=3,color='black',alpha=1,label='1:1')
            axs['histx'].hist(10**a,bins=10 ** np.linspace(minval, maxval, 40),log=True, color='slategray')
            axs['histy'].hist(10**b,bins=10 ** np.linspace(minval, maxval, 40),log=True, color='slategray',orientation='horizontal')
        else:            
            axs['scatter'].plot(np.arange(10**0,10**maxval),np.arange(10**0,10**maxval),linewidth=3,color='black',alpha=1,label='1:1')
            axs['histx'].hist(10**a,bins=10 ** np.linspace(np.log10(1), maxval, 40),log=True, color='slategray')
            axs['histy'].hist(10**b,bins=10 ** np.linspace(np.log10(1), maxval, 40),log=True, color='slategray',orientation='horizontal')
        
        axs['histx'].set_xscale('log')
        axs['histy'].set_yscale('log')

        axs['histx'].tick_params(axis="x", labelbottom=False)
        axs['histy'].tick_params(axis="y", labelleft=False)  
        
        #axs['histx'].set_xlim(right=maxval)
        #axs['histy'].set_ylim(bottom=0.7,top=15000)
    if kind=='lin':
        #no log axes!
        sc = axs['scatter'].scatter(a,b,c=c,s=15,cmap='viridis',vmin=0,vmax=0.025)
        #sc = axs.scatter(a,b,c=c,s=8,cmap='viridis',vmin=np.min(c),vmax=np.max(c))
        
        #axs.plot(np.arange(0,maxval),np.arange(0,maxval),linewidth=3,color='black',alpha=1,label='1:1')
        if variable == 'IWV':
            axs['scatter'].plot(np.arange(0,maxval+5),np.arange(0,maxval+5),linewidth=2,color='black',alpha=1,label='1:1')
            axs['scatter'].set_ylim(bottom=minval,top=maxval)
            axs['scatter'].set_xlim(left=minval,right=maxval)

            axs['histx'].hist(a,bins=np.linspace(minval, maxval, 40),log=True, color='slategray')
            axs['histy'].hist(b,bins=np.linspace(minval, maxval, 40),log=True, color='slategray',orientation='horizontal')
            axs['histy'].set_ylim(bottom=minval,top=maxval)
            axs['histx'].set_xlim(left=minval,right=maxval)
            axs['scatter'].set_xlabel(f'Target {variable} '+'\\ kg m$^{-2}$')
            axs['scatter'].set_ylabel(f'NN Predicted {variable} '+'\\ kg m$^{-2}$')

        else:
            axs['scatter'].plot(np.arange(0,maxval),np.arange(0,maxval),linewidth=3,color='black',alpha=1,label='1:1')
    
            axs['histx'].hist(a,bins=np.linspace(0, maxval, 40),log=True, color='slategray')
            axs['histy'].hist(b,bins=np.linspace(0, maxval, 40),log=True, color='slategray',orientation='horizontal')
            #axs['histx'].set_xscale('log')
        #axs['histy'].set_yscale('log')
    
        axs['histx'].tick_params(axis="x", labelbottom=False)
        axs['histy'].tick_params(axis="y", labelleft=False)  
    if variable != 'IWV':
        #plotlines
        axs['scatter'].plot(hyd_true,RE_20_plus,linewidth=1,linestyle='dotted',color='black',alpha=0.9,label='±20%')
        axs['scatter'].plot(hyd_true,RE_20_minus,linewidth=1,linestyle='dotted',color='black',alpha=0.9)
        axs['scatter'].plot(hyd_true,RE_50_plus,linewidth=1,linestyle='dashed',color='black',alpha=0.9,label='±50%')
        axs['scatter'].plot(hyd_true,RE_50_minus,linewidth=1,linestyle='dashed',color='black',alpha=0.9)
        axs['scatter'].plot(hyd_true,RE_80_plus,linewidth=1,linestyle='dashdot',color='black',alpha=0.9,label='±80%')
        axs['scatter'].plot(hyd_true,RE_80_minus,linewidth=1,linestyle='dashdot',color='black',alpha=0.9)
            
    else:
        axs['scatter'].plot(hyd_true,RE_5_plus,linewidth=2,linestyle='dashed',color='black',alpha=0.9,label='±5%')
        axs['scatter'].plot(hyd_true,RE_5_minus,linewidth=2,linestyle='dashed',color='black',alpha=0.9)
        axs['scatter'].plot(hyd_true,RE_20_plus,linewidth=2,linestyle='dotted',color='black',alpha=0.9,label='±20%')
        axs['scatter'].plot(hyd_true,RE_20_minus,linewidth=2,linestyle='dotted',color='black',alpha=0.9)

    

    
        
    
    cbar_ax = fig.add_axes([0.839, 0.839, 0.02, 0.145])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point density',size=10)
    
    axs['scatter'].legend()
    
    #axs.text(0.02, 0.8, 'Bias: '+str(bias), 
    #         transform=axs.transAxes, fontsize=20,
    #         verticalalignment='top')
    #axs.text(0.05, 0.885, 'RMSE: '+str(rmse), 
    #         transform=axs.transAxes, fontsize=20,
    #         verticalalignment='top')
    #axs.text(0.05, 0.81, 'Corr: '+str(corr), 
    #         transform=axs.transAxes, fontsize=20,
    #         verticalalignment='top')
    
    print('Bias: '+str(bias))
    print('RMSE: '+str(rmse))
    print('Corr: '+str(corr))
    if plotname != None:
        plt.savefig(f'/home/u/u301032/orcestra/plots/target_vs_predicted_scatter_hist_{variable}_{plotname}.jpg', dpi=400)





def plot_scatter_log_v2(hyd_test,hyd_test_predictions_squared, variable, kind = 'log'):
    if kind == 'linear':
        
        a = hyd_test
        b = hyd_test_predictions_squared
        
        a[a<1.] = 0
        b[b<1.] = 0
        
        #a[a<1.] = np.nan
        #b[b<1.] = np.nan
            
    if kind == 'log':
        
        hyd_test_log = hyd_test.copy()
        hyd_test_log[hyd_test_log==0] = 10**(-16)
        #hyd_test_log[hyd_test_log<1] = np.nan
        hyd_test_predictions_squared_log = hyd_test_predictions_squared.copy()
        hyd_test_predictions_squared_log[hyd_test_predictions_squared_log==0] = 10**(-16)
        #hyd_test_predictions_squared_log[hyd_test_predictions_squared_log<1] = np.nan
        
        a = np.log10(hyd_test_log)
        b = np.log10(hyd_test_predictions_squared_log)
        
        a[a<0.] = 0
        b[b<0.] = 0
        
        #a[a<0.] = np.nan
        #b[b<0.] = np.nan
        
        #a = hyd_test.copy()
        #b = hyd_test_predictions_squared.copy()
        
        #a[a<1.] = 0.
        #b[b<1.] = 0.
        
    if kind == 'sqrt':
        
        a = np.sqrt(hyd_test)
        b = hyd_test_predictions
        
    maxval = max(np.nanmax(a),np.nanmax(b))
    minval = max(np.nanmin(a),np.nanmin(b))    
    
    #a[a<0.] = 0
    #b[b<0.] = 0
    
    #f, ax = plt.subplots()
    
    #sns.set(color_codes=True)
    
    jp = sns.jointplot(x = a, y = b,
                       kind = "hist", data = None, cmap='viridis',vmin=0, vmax=60, height=8,)
    if variable == 'IWV':
        jp.ax_joint.plot(np.arange(minval,maxval),np.arange(minval,maxval),linewidth=3,color='black',alpha=1,label='1:1')    
        #jp.ax_joint.plot(np.arange(0,4.5),np.arange(0,4.5),linewidth=3,linestyle='dashed',color='darkred',alpha=0.5)
    else:    
        jp.ax_joint.plot(np.arange(0,maxval),np.arange(0,maxval),linewidth=3,color='black',alpha=1,label='1:1')
        jp.ax_joint.plot(np.arange(0,4.5),np.arange(0,4.5),linewidth=3,linestyle='dashed',color='darkred',alpha=0.5)
    #jp.ax_joint.plot(iwp_true,RE_20_plus,linewidth=2,linestyle='dotted',color='black',alpha=0.9,label='±20%')
    #jp.ax_joint.plot(iwp_true,RE_20_minus,linewidth=2,linestyle='dotted',color='black',alpha=0.9)
    #jp.ax_joint.plot(iwp_true,RE_50_plus,linewidth=2,linestyle='dashed',color='black',alpha=0.9,label='±50%')
    #jp.ax_joint.plot(iwp_true,RE_50_minus,linewidth=2,linestyle='dashed',color='black',alpha=0.9)
    #jp.ax_joint.plot(iwp_true,RE_100_plus,linewidth=2,linestyle='dashdot',color='black',alpha=0.9,label='±100%')
    #jp.ax_joint.plot(iwp_true,RE_100_minus,linewidth=2,linestyle='dashdot',color='black',alpha=0.9)
    #jp.ax_joint.set_xlim(0,maxval)
    #jp.ax_joint.set_ylim(0,maxval)
    #jp.ax_joint.set_xscale('log')
    #jp.ax_joint.set_yscale('log')
    
    if variable == 'IWV':
        jp.ax_joint.set_xlabel(f'True {variable}'+' [kg m$^{-2}$]')
        jp.ax_joint.set_ylabel(f'Retrieved {variable}'+' [kg m$^{-2}$]')
    else:
        jp.ax_joint.set_xlabel(f'True {variable}'+' [g m$^{-2}$]')
        jp.ax_joint.set_ylabel(f'Retrieved {variable}'+' [g m$^{-2}$]')
    #plt.plot(np.arange(0,maxval),np.arange(0,maxval),linewidth=2,color='black',alpha=0.5)
    #plt.xscale('log')
    #plt.yscale('log')
    
    #sns.set(color_codes=False)
    
    #plt.savefig('/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels/plot')

def no_nan_for_plot(targets, predictions):
    nans = np.logical_or(np.isnan(targets), np.isnan(predictions))
    a = targets[~nans]
    b = predictions[~nans]
    return a,b 
def plot_NN_error(true, prediction,variable):
    a,b=no_nan_for_plot(true, prediction)
    # Define bin edges (e.g., 10 bins)
    num_bins = 20
    
    if variable == 'IWV':
        bins = np.linspace(np.min(a), np.max(a), num_bins + 1)
    else: 
        bin_max = 3 # max(a) #set to number of interest or maybe with if variable == 'IWP':
        bin_min = 0
        num_bins = (bin_max - bin_min) *10 
        bins  = np.logspace(bin_min, bin_max, num_bins + 1)
    # 
    
    # Digitize 'a' to find out which bin each value belongs to
    bin_indices = np.digitize(b, bins)
    
    # Initialize arrays to hold results
    bin_centers = (bins[:-1] + bins[1:]) / 2
    b_error = np.empty(num_bins)
    rel_100 = np.empty(num_bins)
    # Calculate error (e.g., standard deviation) of 'b' within each bin
    for i in range(1, num_bins + 1):
        # Find indices of data points in the current bin
        in_bin = bin_indices == i
        #error as rmse between true and predicted # like in marek s paper
        targets= a[in_bin]
        predictions= b[in_bin]
        b_error[i-1]=np.sqrt(((predictions - targets) ** 2).mean())
        ## Calculate standard deviation of 'b' for these points
        #b_error[i - 1] = np.std(b[in_bin])
        #rel_100=
    # Plot

    plt.plot(bin_centers,0.2*bin_centers,linestyle='-',color='grey') #relative erorr 100 percent
    plt.plot(bin_centers,0.1*bin_centers,linestyle=':',color='grey') #relative erorr 100 percent
    plt.plot(bin_centers,b_error)
    
    #plt.errorbar(bin_centers, np.zeros_like(bin_centers), yerr=b_error, fmt='o', capsize=5)

    if variable == 'IWV':
        plt.xlabel(variable+'[kg m$^{-2}$]')
        plt.plot(bin_centers,0.05*bin_centers,linestyle='--',color='grey') #relative erorr 100 percent
        plt.plot(bin_centers,0.01*bin_centers,linestyle='-.',color='grey') #relative erorr 100 percent
    else:
        plt.plot(bin_centers,bin_centers,linestyle='--',color='grey') #relative erorr 100 percent
        plt.plot(bin_centers,0.5*bin_centers,linestyle='-.',color='grey') #relative erorr 100 percent
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('retrieved '+variable+' [g m$^{-2}$]')
        
        if  variable == 'IWP':
            plt.ylim(top=300)
        else:
            plt.ylim(top=120)       
    
    plt.ylabel('Error retrieved - True')
    #plt.title('Error of b within each a bin')
    plt.show()

def plot_NN_error_v2(true, prediction,variable,filename=None):
    #so far used for graphics
    a,b=no_nan_for_plot(true, prediction)
    # Define bin edges (e.g., 10 bins)
    num_bins = 20
    
    if variable == 'IWV':
        bins = np.linspace(np.min(a), np.max(a), num_bins + 1)
    else: 
        if variable == 'IWP':
            bin_max = 4 # max(a) #set to number of interest or maybe with 
        if variable == 'LWP':
            bin_max = 4
        bin_min = 0
        num_bins = (bin_max - bin_min) *4
        bins  = np.logspace(bin_min, bin_max, num_bins + 1)
    # 
    
    # Digitize 'a' to find out which bin each value belongs to
    bin_indices = np.digitize(b, bins)
    
    # Initialize arrays to hold results
    bin_centers = (bins[:-1] + bins[1:]) / 2
    b_error = np.empty(num_bins)
    rel_100 = np.empty(num_bins)
    # Calculate error (e.g., standard deviation) of 'b' within each bin
    for i in range(1, num_bins + 1):
        # Find indices of data points in the current bin
        in_bin = bin_indices == i
        #error as rmse between true and predicted # like in marek s paper
        targets= a[in_bin]
        predictions= b[in_bin]
        b_error[i-1]=np.sqrt(((targets -predictions)**2).mean())
        ## Calculate standard deviation of 'b' for these points
        #b_error[i - 1] = np.std(b[in_bin])
        #rel_100=
    # Plotting
    fig, axs = plt.subplots(figsize=(5,6))
    
    # Plot Error curves
    
    x=np.logspace(bin_min, bin_max, num_bins *10)
    if variable == 'IWV':
        error_curves =[1,5,10,25]
    else:
        error_curves =[10,25,50,100]
    
    for e in error_curves:
        
        plt.plot(x,(0.01*e)*x,linestyle=':',color='grey',label=e) #relative erorr 20 percent
        X=1200#1200#400 + (0.01*e)*800
        plt.text(X*(1/(0.01*e)), X,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=80)
        #plt.scatter(X*(1/(0.01*e)), X)

        
    #plt.plot(x,0.15*x,linestyle=':',color='grey',label='15') #relative erorr 10 percent
    
    plt.plot(bin_centers,b_error)
    
    #plt.errorbar(bin_centers, np.zeros_like(bin_centers), yerr=b_error, fmt='o', capsize=5)

    if variable == 'IWV':
        plt.xlabel('Retrieved '+ variable+'[kg m$^{-2}$]')
        plt.plot(bin_centers,0.05*bin_centers,linestyle='--',color='grey') #relative erorr 100 percent
        plt.plot(bin_centers,0.01*bin_centers,linestyle='-.',color='grey') #relative erorr 100 percent
    else:
        plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('Retrieved '+ variable+' \\ g m$^{-2}$')
        
        if  variable == 'IWP':
            plt.ylim(bottom=-50,top=1300)
        if  variable == 'LWP':
            plt.ylim(bottom=-50,top=1300)
        else:
            plt.ylim(top=120)       

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.ylabel('Error: True - retrieved '+variable+' \\ g m$^{-2}$')
    #plt.title('Error of b within each a bin')
    plt.tight_layout()
    if filename != None:
        plt.savefig(f'/home/u/u301032/orcestra/plots/{variable}_error_NN_{filename}.png',dpi=400)
    plt.show()

def plot_NN_error_v2_with_bias(true, prediction,variable,filename=None):
    #v2 but now with bias and std
    a,b=no_nan_for_plot(true, prediction)
    # Define bin edges (e.g., 10 bins)
    num_bins = 20
    
    if variable == 'IWV':
        bins = np.linspace(np.min(a), np.max(a), num_bins + 1)
    else: 
        if variable == 'IWP':
            bin_max = 4 # max(a) #set to number of interest or maybe with 
        if variable == 'LWP':
            bin_max = 4
        bin_min = 0
        num_bins = (bin_max - bin_min) *4
        bins  = np.logspace(bin_min, bin_max, num_bins + 1)
    # 
    
    # Digitize 'a' to find out which bin each value belongs to
    bin_indices = np.digitize(b, bins)
    
    # Initialize arrays to hold results
    bin_centers = (bins[:-1] + bins[1:]) / 2
    b_error = np.empty(num_bins)
    b_bias = np.empty(num_bins)
    b_std = np.empty(num_bins)
    
    rel_100 = np.empty(num_bins)
    # Calculate error (e.g., standard deviation) of 'b' within each bin
    for i in range(1, num_bins + 1):
        # Find indices of data points in the current bin
        in_bin = bin_indices == i
        #error as rmse between true and predicted # like in marek s paper
        targets= a[in_bin]
        predictions= b[in_bin]
        b_error[i-1]= np.sqrt(((targets -predictions)**2).mean())
        b_bias[i-1]= (predictions - targets).mean()
        ## Calculate standard deviation of 'b' for these points
        b_std[i - 1] = np.std(predictions - b_bias[i-1])
        #rel_100=
    # Plotting
    fig, axs = plt.subplots(figsize=(5,6))
    
    # Plot Error curves
    
    x=np.logspace(bin_min, bin_max, num_bins *10)
    if variable == 'IWV':
        error_curves =[1,5,10,25]
    else:
        error_curves =[10,25,50,100]
    
    for e in error_curves:
        
        plt.plot(x,(0.01*e)*x,linestyle=':',color='grey',label=e) #relative erorr 20 percent
        X=1200#1200#400 + (0.01*e)*800
        plt.text(X*(1/(0.01*e)), X,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=80)
        #plt.scatter(X*(1/(0.01*e)), X)

        
    #plt.plot(x,0.15*x,linestyle=':',color='grey',label='15') #relative erorr 10 percent
    
    plt.plot(bin_centers,b_error)
    plt.plot(bin_centers,b_bias)
    #plt.errorbar(bin_centers, np.zeros_like(bin_centers), yerr=b_error, fmt='o', capsize=5)

    if variable == 'IWV':
        plt.xlabel('Retrieved '+ variable+'[kg m$^{-2}$]')
        plt.plot(bin_centers,0.05*bin_centers,linestyle='--',color='grey') #relative erorr 100 percent
        plt.plot(bin_centers,0.01*bin_centers,linestyle='-.',color='grey') #relative erorr 100 percent
    else:
        plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('Retrieved '+ variable+' \\ g m$^{-2}$')
        
        if  variable == 'IWP':
            plt.ylim(bottom=-150,top=1300)
        else:
            plt.ylim(top=120)       

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.ylabel('Error: True - retrieved '+variable+' \\ g m$^{-2}$')
    #plt.title('Error of b within each a bin')
    plt.tight_layout()
    if filename != None:
        plt.savefig(f'/home/u/u301032/orcestra/plots/{variable}_error_NN_{filename}.png',dpi=400)
    plt.show()
    print(bin_centers,b_bias)
    plt.plot(bin_centers,b_bias)
    plt.plot(bin_centers,b_bias+b_std,c='r')
    plt.plot(bin_centers,b_bias-b_std,c='r')
    #plt.yscale('log')
    plt.xscale('log')
    plt.show()
    #return bin_centers,b_bias
    
def plot_NN_error_v3(true, prediction,variable,filename=None):
    # more in the version of brath
    print(np.isnan(true).sum(),np.isnan(prediction).sum())
    a,b=no_nan_for_plot(true, prediction)
    print(np.isnan(a).sum(),np.isnan(b).sum())
    a[a<=0]=10*6
    
    b[b<=0]=10*6
    print(np.isnan(a).sum(),np.isnan(b).sum())
    a=np.log10(a)
    b=np.log10(b)
    print(np.isnan(a).sum(),np.isnan(b).sum())
    num_bins = 20
    plt.hist([a,b],bins=80,log=True)
    plt.show()

    # Define bin edges (e.g., 10 bins)
    if variable == 'IWV':
        bins = np.linspace(np.min(a), np.max(a), num_bins + 1)
    else: 
        if variable == 'IWP':
            bin_max = 4 # max(a) #set to number of interest or maybe with 
        if variable == 'LWP':
            bin_max = 4
        bin_min = -10
        num_bins = (bin_max - bin_min) *4
        bins  = np.logspace(bin_min, bin_max, num_bins + 1)
        bins  = np.linspace(bin_min, bin_max, num_bins + 1)
    # 


    
    # Digitize 'a' to find out which bin each value belongs to
    bin_indices = np.digitize(b, bins)
    
    # Initialize arrays to hold results
    bin_centers = (bins[:-1] + bins[1:]) / 2
    b_error = np.empty(num_bins)
    rel_100 = np.empty(num_bins)
    b_bias= np.empty(num_bins)
    b_std= np.empty(num_bins)
    n = np.empty(num_bins)
    # Calculate error (e.g., standard deviation) of 'b' within each bin
    for i in range(1, num_bins + 1):
        # Find indices of data points in the current bin
        in_bin = bin_indices == i
        #error as rmse between true and predicted # like in marek s paper
        targets= a[in_bin]
        predictions= b[in_bin]
        #b_error[i-1]=((predictions-targets)/predictions).mean()#((predictions-targets)/predictions).mean()
        b_error[i-1]= np.sqrt(((targets -predictions)**2).mean())
        ## Calculate standard deviation of 'b' for these points
        n=len(in_bin)
        b_bias[i-1]= (predictions - targets).mean()
        ## Calculate standard deviation of 'b' for these points
        b_std[i - 1] = np.sqrt(((predictions - b_bias[i-1])**2).mean())#np.std(predictions - b_bias[i-1])
    # Plotting
    fig, ax = plt.subplots()
    #ax.pcolormesh(bins,bins,n)
    sns.histplot(x = b, y = (b-a),bins=120)#,kws=dict(bins=30))#,   kind = "hist", data = None, cmap='viridis',marginal_kws=dict(bins=30),)#,vmin=0, vmax=60, height=8,)
    plt.xlim(-1,5)
    plt.ylim(-5,5)
    plt.plot(bin_centers,b_error)

    
    plt.plot(bin_centers,b_bias+b_std,c='r')
    plt.plot(bin_centers,b_bias-b_std,c='r')
    plt.plot(bin_centers,b_bias)
    print(b_error,b_bias,b_std)

    
    plt.xlabel('retrieved '+variable+' [g m$^{-2}$]')
    plt.ylabel('Error: True - retrieved '+variable+' \\ g m$^{-2}$')
    plt.show()
    
    fig, axs = plt.subplots(figsize=(10,6))
    
    # Plot Error curves
    
    x=np.logspace(bin_min, bin_max, num_bins *10)
    if variable == 'IWV':
        error_curves =[1,5,10,25]
    else:
        error_curves =[10,25,50,100]
    
    for e in error_curves:
        
        #plt.plot(x,(0.01*e)*x,linestyle=':',color='grey',label=e) #relative erorr 20 percent
        X=1200#400 + (0.01*e)*800
        #plt.text(X*(1/(0.01*e)), X,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=80)
        #plt.scatter(X*(1/(0.01*e)), X)

        
    #plt.plot(x,0.15*x,linestyle=':',color='grey',label='15') #relative erorr 10 percent
    
    
    #plt.errorbar(bin_centers, np.zeros_like(bin_centers), yerr=b_error, fmt='o', capsize=5)

    if variable == 'IWV':
        plt.xlabel(variable+'[kg m$^{-2}$]')
        plt.plot(bin_centers,0.05*bin_centers,linestyle='--',color='grey') #relative erorr 100 percent
        plt.plot(bin_centers,0.01*bin_centers,linestyle='-.',color='grey') #relative erorr 100 percent
    else:
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('True '+ variable+' \\ g m$^{-2}$')
            

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.ylabel('Error: True - retrieved '+variable+' \\ g m$^{-2}$')
    #plt.title('Error of b within each a bin')
    plt.tight_layout()
    if filename != None:
        plt.savefig(f'/home/u/u301032/orcestra/plots/{variable}_error_NN_{filename}.png',dpi=400)



def plot_and_calculate_NN_bias(true, prediction,variable,filename=None):
    #can be used to clculate bias for custom maed bins eg no clouds vs all clouds
    a,b=no_nan_for_plot(true, prediction)
    # Define bin edges (e.g., 10 bins)
    num_bins = 20
    
    if variable == 'IWV':
        bins = np.linspace(np.min(a), np.max(a), num_bins + 1)
    else: 
        if variable == 'IWP':
            bin_max = 4 # max(a) #set to number of interest or maybe with 
        if variable == 'LWP':
            bin_max = 4
        bin_min = 0
        num_bins = (bin_max - bin_min) *4
        bins  = np.logspace(bin_min, bin_max, num_bins + 1)
        bins= np.array([10**(-8),1,10,100,1000,10000,100000])
        num_bins = len(bins)-1
    # Digitize 'a' to find out which bin each value belongs to
    b[b<=10**(-6)] =10**(-6)
    bin_indices = np.digitize(b, bins)
    
    # Initialize arrays to hold results
    bin_centers = (bins[:-1] + bins[1:]) / 2
   
    b_bias= np.empty(num_bins)

    # Calculate error (e.g., standard deviation) of 'b' within each bin
    for i in range(1, num_bins + 1):
        # Find indices of data points in the current bin
        in_bin = bin_indices == i
        #error as rmse between true and predicted # like in marek s paper
        targets= a[in_bin]
        predictions= b[in_bin]

        b_bias[i-1]= (predictions - targets).mean()

    plt.plot(bin_centers,b_bias)
    plt.ylim(bottom=-50,top=2)
    plt.xscale('log')
    
    plt.show()
    return bins,bin_centers,b_bias
    