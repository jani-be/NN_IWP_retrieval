"""
Plots additional to NN training
 """


#%%
#%% Loading packages
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats
import fsspec
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import glob
from functools import partial
import sys
sys.path.append('/home/u/u301032/orcestra/NN_IWP_retrieval/NN_training_and_development/')
print(sys.path)
from src import retrieval_dev_fcts as dev
from src import retrieval_plots as rp
#sys.path.append('/home/u/u301238/master_thesis/')
sys.path.append('/home/u/u301032/orcestra/NN_IWP_retrieval/')
#import src
#import src_comparison_halo_pamtra as chp

import matplotlib.colors as colors
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature


# %%
training_version = "_v2" #''
## LOAD teat and predicted data
IWV_test_predictions_squared=np.load(f'/work/um0203/u301032/master_thesis/retrieval_test_data/IWV_test_predictions_squared{training_version}.npy')
IWP_test_predictions_squared=np.load(f'/work/um0203/u301032/master_thesis/retrieval_test_data/IWP_test_predictions_squared{training_version}.npy')
#LWP_test_predictions_squared=np.load(f'/work/um0203/u301032/master_thesis/retrieval_test_data/LWP_test_predictions_squared{training_version}.npy')
LWP_test_predictions_squared=np.load(f'/work/um0203/u301032/master_thesis/retrieval_test_data/LWP_test_predictions_squared.npy')
CLWP_test_predictions_squared=np.load(f'/work/um0203/u301032/master_thesis/retrieval_test_data/CLWP_test_predictions_squared{training_version}.npy')
#test_LWP=np.load(f'/work/um0203/u301032/master_thesis/retrieval_test_data/test_LWP{training_version}.npy')
test_LWP=np.load(f'/work/um0203/u301032/master_thesis/retrieval_test_data/test_LWP.npy')
test_CLWP=np.load(f'/work/um0203/u301032/master_thesis/retrieval_test_data/test_CLWP{training_version}.npy')
test_IWP=np.load(f'/work/um0203/u301032/master_thesis/retrieval_test_data/test_IWP{training_version}.npy')
test_IWV=np.load(f'/work/um0203/u301032/master_thesis/retrieval_test_data/test_IWV{training_version}.npy')

data=[[test_IWV,IWV_test_predictions_squared],[test_CLWP,CLWP_test_predictions_squared],[test_IWP,IWP_test_predictions_squared]]
variables=['IWV','LWP','CLWP','IWP']
target_data = {
    'IWV': test_IWV,#[::100],
    'LWP': test_LWP,#[::100],
    'CLWP': test_CLWP,#[::100],
    'IWP': test_IWP,#[::100]

}
prediction_data = {
    'IWV': IWV_test_predictions_squared,#[::100],
    'LWP': LWP_test_predictions_squared,#[::100],
    'CLWP': CLWP_test_predictions_squared,#[::100],
    'IWP': IWP_test_predictions_squared,#[::100]
}
#%%

#%%
variables=['IWV','LWP','CLWP','IWP']





#%%

def plot_scatter_row(filename=None):
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




#%%
fig = plt.figure(figsize=(12, 12),constrained_layout=True)

subfigs = fig.subfigures(2, 2)
subfigs[0,1]

for outerind, subfig in enumerate(subfigs.flat):
    if subfig ==subfigs[0,1]:
        ax=subfig.subplots()
        ax.axis('off')
    else:
        subfig.suptitle(f'Subfig {outerind}')

        axs = subfig.subplot_mosaic([['histx', '.'],
                                ['scatter', 'histy']],
                                 width_ratios=(5, 1), height_ratios=(1, 5))

plt.show()
#%%
def plot_2_by_1_errors_MFE(target_data, prediction_data, variables=['CLWP','IWP'],ylim=[0,600], filename=None):

    # Erstellen der Figure mit 3 Unterplots in einer Spalte
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))#15,5
    
    # Für jedes Variable: plot auf dem jeweiligen Subplot
    for ax, var, title in zip(axes, variables, ['a)','b)']):
        # Aufrufen der angepassten Funktion

        plot_NN_error_v4(target_data[var], prediction_data[var], var, filename=filename,error='MFE', ax=ax,ylim=ylim)
        # Optional: Titel oder andere Anpassungen
        
        ax.text(0.05, 1.03, title, transform=ax.transAxes,
          fontsize=16,  va='top')
    
    plt.tight_layout()
    if filename:
        plt.savefig(f'/home/u/u301032/orcestra/plots/error_MFE_2x1_{filename}.png')
        print('Plot gespeichert als:', f'/home/u/u301032/orcestra/plots/error_MFE_2x1_{filename}.png')
    plt.show()

#%%
def plot_3_by_1_errors(target_data, prediction_data, variables=['IWV','CLWP','IWP'],ylim=[-30,350], filename=None):

    # Erstellen der Figure mit 3 Unterplots in einer Spalte
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))#15,5
    
    # Für jedes Variable: plot auf dem jeweiligen Subplot
    for ax, var, title in zip(axes, variables, ['a)','b)','c)']):
        # Aufrufen der angepassten Funktion

        plot_NN_error_v4(target_data[var], prediction_data[var], var, filename=filename, ax=ax,ylim=ylim)
        # Optional: Titel oder andere Anpassungen
        
        ax.text(0.05, 1.03, title, transform=ax.transAxes,
          fontsize=16,  va='top')
    
    plt.tight_layout()
    if filename:
        plt.savefig(f'/home/u/u301032/orcestra/plots/RMSEandbias_3x1{filename}.png')
        print('Plot gespeichert als:', f'/home/u/u301032/orcestra/plots/RMSEandbias_3x1{filename}.png')
    plt.show()
#%%
dev.plot_3_by_1_errors(target_data,prediction_data)

dev.plot_NN_error_v4(test_IWV,IWV_test_predictions_squared,variable='IWV')    
# %%


# %%

def calc_bias(targets,predictions,bin=[0,5]):
    mask=np.logical_and(predictions>= bin[0], predictions<= bin[1])
    print(mask.sum())
    p=predictions[mask]
    t=targets[mask]
    bias = ( t - p).mean()
    return bias
calc_bias(test_CLWP,CLWP_test_predictions_squared,bin=[0,1])
# %%
plot_3_by_1_errors(target_data,prediction_data,variables=['IWV','CLWP','IWP'],filename=None)#'v2_newrun'#'v1'

# %%

def plot_NN_error_v4(true, prediction,variable,ax=None,filename=None,error='RMSE',ylim=[-50,80]):
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
            bin_max = 4 # max(a) #set to number of interest or maybe with 
        if variable == 'LWP':
            bin_max = 4
        
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
    b_std2 = np.empty(num_bins)
    b_MFE = np.empty(num_bins)
    b_MFE2 = np.empty(num_bins)
    rel_100 = np.empty(num_bins)
    # Calculate error (e.g., standard deviation) of 'b' within each bin
    for i in range(1, num_bins + 1):
        
        # Find indices of data points in the current bin
        in_bin = bin_indices == i

        if np.sum(in_bin) <=5:
            b_error[i-1]=np.nan
            b_bias[i-1]=np.nan
            b_std[i - 1]=np.nan
            b_MFE[i - 1]=np.nan
            print(np.log10(bin_centers[i-1]))
            continue
        #error as rmse between true and predicted # like in marek s paper
        targets= a[in_bin]
        predictions= b[in_bin]
        b_error[i-1]= np.sqrt(((targets-predictions)**2).mean())
        #rel_100[i-1]= np.sqrt((((predictions - targets)/predictions)**2).mean())
        b_bias[i-1]= ( targets - predictions).mean()
        ## Calculate standard deviation of 'b' for these points
        b_std[i - 1] = np.std(predictions - targets)
        #b_std2[i - 1] = np.sqrt(((predictions - targets-b_bias[i-1])**2).mean())
        #rel_100=
        b_MFE[i - 1] = np.median(10**(np.abs(np.log10(predictions/targets)))-1)*100
        #b_MFE2[i - 1] = np.median(10**(np.abs(predictions-targets)))*100
        print(np.sum(in_bin),np.round((bin_centers[i-1])),'bias: ',np.round(b_bias[i-1]),np.round(b_bias[i-1]/(bin_centers[i-1])*100),'RMSE: ',np.round(b_error[i-1]),np.round(b_error[i-1]/(bin_centers[i-1])*100))
    
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,6))  
        show_plot = True
    else:
        show_plot = False
    # Plot Error curves
    
    


    match error:
        case 'RMSE':

            #ax.plot(bin_centers,b_std,label='std')
            if variable == 'IWV':
                error_curves =[-1,1,2,5]
                x=np.linspace(np.min(b)-1, np.max(b)+3,200)
            else:
                error_curves =[5,10,25,50,100,250,500]
                x=np.logspace(bin_min, bin_max, num_bins *10)
            for e in error_curves:
                
                ax.plot(x,(0.01*e)*x,linestyle=':',color='grey',label=e) #relative erorr 20 percent
                if variable=='IWV':
                    X=np.max(b)#-0.05*(np.max(b)-np.min(b))
                    Y=X*((0.01*e))
                    print(e,Y,X)
                    #plt.scatter(X, Y)
                    ax.text(X, Y,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=0)
                    #ax.text(X, -Y,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=0)
                    #ax.plot(x,-(0.01*1)*x,linestyle=':',color='grey',label=e)
                else:
                    ax.plot(x,-(0.01*e)*x,linestyle=':',color='grey',label=e)
                    X=ylim[1]-0.05*(ylim[1]-ylim[0])#950#1200#400 + (0.01*e)*800
                    ax.text(X*(1/(0.01*e)), X,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=80)
                    #plt.scatter(X*(1/(0.01*e)), X)
            ax.plot(x,0*x,linestyle='-',color='grey',label=e)
            ax.plot(bin_centers,b_error,label='RMSE')
            ax.plot(bin_centers,b_bias,label='bias')
            ax.grid()
        case 'MFE':
            ax.plot(bin_centers,b_MFE,label='MFE')
            ax.grid()
          
    
    #plt.errorbar(bin_centers, np.zeros_like(bin_centers), yerr=b_error, fmt='o', capsize=5)

    if variable == 'IWV':
        ax.set_xlabel('Retrieved '+ variable+' \\ kg m$^{-2}$')
        ax.set_ylabel('Error: True - retrieved  '+variable+' \\ kg m$^{-2}$')
        ax.set_ylim(bottom=-1,top=5)
        
        #plt.plot(bin_centers,0.05*bin_centers,linestyle='--',color='grey') #relative erorr 100 percent
        #plt.plot(bin_centers,0.01*bin_centers,linestyle='-.',color='grey') #relative erorr 100 percent
    else:
        ax.set_ylabel('Error: True - retrieved  '+variable+' \\ g m$^{-2}$')
        ax.set_xscale('log')
        #plt.yscale('log')
        ax.set_xlabel('Retrieved '+ variable+' \\ g m$^{-2}$')
        
        if  variable == 'IWP':
            ax.set_ylim(bottom=ylim[0],top=ylim[1])   #(bottom=-320,top=1020) #
        elif  variable == 'LWP':
            ax.set_ylim(bottom=ylim[0],top=ylim[1])   #(bottom=-220,top=1000) 
            1
            #plt.yscale('log')
        else:
            ax.set_ylim(top=120)    
            ax.set_ylim(bottom=ylim[0],top=ylim[1])   

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #plt.title('Error of b within each a bin')
    
    if filename and show_plot:
        plt.tight_layout()
        plt.savefig(f'/home/u/u301032/orcestra/plots/{variable}_error_biasRMSE_{filename}.png',dpi=400)
    if show_plot:
        plt.tight_layout()
        plt.show()

# %%