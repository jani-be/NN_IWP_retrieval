# LOAD functions
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
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


# These functions are used to plot training and testing data, as well as errors from the retrieval


def plot_scatter_log_v2(hyd_test,hyd_test_predictions_squared, variable, kind = 'log'):
           
    if kind == 'log':
        
        hyd_test_log = hyd_test.copy()
        hyd_test_log[hyd_test_log==0] = 10**(-16)
        #hyd_test_log[hyd_test_log<1] = np.nan
        hyd_test_predictions_squared_log = hyd_test_predictions_squared.copy()
        hyd_test_predictions_squared_log[hyd_test_predictions_squared_log==0] = 10**(-16)
        #hyd_test_predictions_squared_log[hyd_test_predictions_squared_log<1] = np.nan
        
        a = np.log10(hyd_test_log)
        b = np.log10(hyd_test_predictions_squared_log)
        
        #a[a<0.] = 0
        #b[b<0.] = 0
        
        #a[a<0.] = np.nan
        #b[b<0.] = np.nan
        
        #a = hyd_test.copy()
        #b = hyd_test_predictions_squared.copy()
        
        #a[a<1.] = 0.
        #b[b<1.] = 0.
        
        
    maxval = max(np.nanmax(a),np.nanmax(b))
        
    a[a<0.] = 0
    b[b<0.] = 0
    
    #f, ax = plt.subplots()
    
    #sns.set(color_codes=True)
    
    jp = sns.jointplot(x = a, y = b,
                       kind = "hist", data = None, cmap='viridis',vmin=0, vmax=60, height=8,joint_kws=dict(bins=80))
    jp.ax_joint.plot(np.arange(0,maxval),np.arange(0,maxval),linewidth=3,color='black',alpha=1,label='1:1')
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
    jp.ax_joint.plot(np.arange(0,4.5),np.arange(0,4.5),linewidth=3,linestyle='dashed',color='darkred',alpha=0.5)
    jp.ax_joint.set_xlabel(f'True {variable}'+' [g m$^{-2}$]')
    jp.ax_joint.set_ylabel(f'Retrieved {variable}'+' [g m$^{-2}$]')
    #plt.plot(np.arange(0,maxval),np.arange(0,maxval),linewidth=2,color='black',alpha=0.5)
    #plt.xscale('log')
    #plt.yscale('log')




def plot_scatter_log( test_hyd,hyd_test_predictions_squared, variable,kind = 'log',plotname=None):
    
    
    #fig, axs = plt.subplots(figsize=(9,8))
    fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['scatter', 'histy']],
                              figsize=(6, 6),
                              width_ratios=(5, 1), height_ratios=(1, 5),
                              layout='constrained')

        
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
        sc = axs['scatter'].scatter(10**a,10**b)#,c=c,s=15,cmap='viridis',vmin=0,vmax=0.3)
        #sc = axs.scatter(a,b,c=c,s=8,cmap='viridis',vmin=np.min(c),vmax=np.max(c))
        
        
        #axs.plot(np.arange(0,maxval),np.arange(0,maxval),linewidth=3,color='black',alpha=1,label='1:1')

        
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
        #axs.set_ylim(bottom=1)
        #axs.set_xlim(left=1)
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


def plot_NN_error_v2(true, prediction,variable,filename=None):
    #so far used for graphics
    a,b=dev.no_nan_for_plot(true, prediction)
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


def plot_NN_error_v4(true, prediction,variable,filename=None):
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
        b_error[i-1]= np.sqrt(((predictions - targets)**2).mean())
        #rel_100[i-1]= np.sqrt((((predictions - targets)/predictions)**2).mean())
        b_bias[i-1]= (predictions - targets).mean()
        ## Calculate standard deviation of 'b' for these points
        #b_std[i - 1] = np.std(predictions - b_bias[i-1])
        #rel_100=

        print(np.sum(in_bin))
    
    # Plotting
    fig, axs = plt.subplots(figsize=(5,6))
    
    # Plot Error curves
    
    
    if variable == 'IWV':
        error_curves =[1,5,10]
        x=np.linspace(np.min(a), np.max(a),200)
    else:
        error_curves =[5,10,25,50,100]
        x=np.logspace(bin_min, bin_max, num_bins *10)
    for e in error_curves:
        plt.plot(x,-(0.01*e)*x,linestyle=':',color='grey',label=e)
        plt.plot(x,(0.01*e)*x,linestyle=':',color='grey',label=e) #relative erorr 20 percent
        if variable=='IWV':
            X=np.min(b)+1
            Y=X*((0.01*e))
            print(e,Y,X)
            #plt.scatter(X, Y)
            plt.text(X, Y,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=0)
        else:
            X=950#1200#400 + (0.01*e)*800
            plt.text(X*(1/(0.01*e)), X,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=80)
            #plt.scatter(X*(1/(0.01*e)), X)
    plt.plot(x,0*x,linestyle='-',color='grey',label=e)

        
    plt.plot(bin_centers,b_error)
    plt.plot(bin_centers,b_bias)
    #plt.errorbar(bin_centers, np.zeros_like(bin_centers), yerr=b_error, fmt='o', capsize=5)

    if variable == 'IWV':
        plt.xlabel('Retrieved '+ variable+' \\ kg m$^{-2}$')
        plt.ylabel('Error: Retrieved - true '+variable+' \\ kg m$^{-2}$')
        plt.ylim(bottom=-1,top=7)
        
        #plt.plot(bin_centers,0.05*bin_centers,linestyle='--',color='grey') #relative erorr 100 percent
        #plt.plot(bin_centers,0.01*bin_centers,linestyle='-.',color='grey') #relative erorr 100 percent
    else:
        plt.ylabel('Error: Retrieved - true '+variable+' \\ g m$^{-2}$')
        plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('Retrieved '+ variable+' \\ g m$^{-2}$')
        
        if  variable == 'IWP':
            plt.ylim(bottom=-220,top=1000) #(bottom=-320,top=1020) #
        elif  variable == 'LWP':
            plt.ylim(bottom=-420,top=1020) #(bottom=-220,top=1000) 
            #plt.yscale('log')
        else:
            plt.ylim(top=120)       

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    
    #plt.title('Error of b within each a bin')
    plt.tight_layout()
    if filename != None:
        plt.savefig(f'/home/u/u301032/orcestra/plots/{variable}_error_biasRMSE_{filename}.png',dpi=400)
    plt.show()
    print(bin_centers,b_bias)

def plot_NN_bias(true, prediction,variable,filename=None):
    #so far used for graphics
    a,b=dev.no_nan_for_plot(true, prediction)
    a[a<10**(-8)] = 10**(-8)
    b[b<10**(-8)] = 10**(-8)
    
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
        #bins = np.array([[0]+list(np.logspace(0,4,16))])
    # 
    
    # Digitize 'a' to find out which bin each value belongs to
    bin_indices = np.digitize(b, bins)
    
    # Initialize arrays to hold results
    bin_centers = (bins[:-1] + bins[1:]) / 2
    b_error = np.empty(num_bins)
    b_bias = np.empty(num_bins)

    # Calculate error (e.g., standard deviation) of 'b' within each bin
    for i in range(1, num_bins + 1):
        # Find indices of data points in the current bin
        in_bin = bin_indices == i
        #error as rmse between true and predicted # like in marek s paper
        targets= a[in_bin]
        predictions= b[in_bin]

        
        b_bias[i-1]= (np.log10(predictions) - np.log10(targets)).mean()

    
    # Plotting
    fig, axs = plt.subplots(figsize=(5,6))
    
    # Plot Error curves
    
    x=np.logspace(bin_min, bin_max, num_bins *10)
    if variable == 'IWV':
        error_curves =[1,5,10,25]
    else:
        error_curves =[5,10,25,50,100]
    
    for e in error_curves:
        #plt.plot(x,-(0.01*e)*x,linestyle=':',color='grey',label=e)
        #plt.plot(x,(0.01*e)*x,linestyle=':',color='grey',label=e) #relative erorr 20 percent
        X=1200#1200#400 + (0.01*e)*800
        #plt.text(X*(1/(0.01*e)), X,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=80)
        #plt.scatter(X*(1/(0.01*e)), X)
    plt.plot(x,0*x,linestyle='-',color='grey',label=e)
    #plt.plot(bin_centers,100*my_RMSE,label='RMSE')  
    #plt.plot(bin_centers,100*RRMSE,label='RMSE')  
    #plt.plot(bin_centers,MRE,label='MRE')
    #plt.plot(bin_centers,MFE,label='MFE')
    #plt.plot(bin_centers,b_error)
    plt.plot(bin_centers,b_bias)
    #plt.errorbar(bin_centers, np.zeros_like(bin_centers), yerr=b_error, fmt='o', capsize=5)
    plt.legend()
    if variable == 'IWV':
        plt.xlabel('Retrieved '+ variable+'[kg m$^{-2}$]')
        plt.plot(bin_centers,0.05*bin_centers,linestyle='--',color='grey') #relative erorr 100 percent
        plt.plot(bin_centers,0.01*bin_centers,linestyle='-.',color='grey') #relative erorr 100 percent
    else:
        plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('Retrieved '+ variable+' \\ g m$^{-2}$')
        
        #if  variable == 'LWP':
            
    #plt.ylim(top=3)
        #elif  variable == 'LWP':
            #plt.ylim(bottom=-150,top=1300)
            #plt.yscale('log')
        #else:
            #plt.ylim(top=120)       

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.ylabel('Bias')
    #plt.title('Error of b within each a bin')
    plt.tight_layout()
    if filename != None:
        plt.savefig(f'/home/u/u301032/orcestra/plots/{variable}_error_NN_{filename}.png',dpi=400)
    plt.show()
    print(bin_centers,b_bias)

def plot_NN_RMSE_error(true, prediction,variable,filename=None):
    #so far used for graphics
    a,b=dev.no_nan_for_plot(true, prediction)

    
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
        #bins = np.array([[0]+list(np.logspace(0,4,16))])
    # 
    
    # Digitize 'a' to find out which bin each value belongs to
    bin_indices = np.digitize(b, bins)
    
    # Initialize arrays to hold results
    bin_centers = (bins[:-1] + bins[1:]) / 2
    b_error = np.empty(num_bins)
    b_bias = np.empty(num_bins)
    b_std = np.empty(num_bins)
    my_RMSE = np.empty(num_bins)
    RRMSE = np.empty(num_bins)
    MFE =  np.empty(num_bins)
    MRE =  np.empty(num_bins)
    # Calculate error (e.g., standard deviation) of 'b' within each bin
    for i in range(1, num_bins + 1):
        # Find indices of data points in the current bin
        in_bin = bin_indices == i
        #error as rmse between true and predicted # like in marek s paper
        targets= a[in_bin]
        predictions= b[in_bin]
        b_error[i-1]= np.sqrt(((predictions - targets)**2).mean())
        RRMSE[i-1]= np.sqrt(((((predictions - targets))**2).mean()/np.sum(predictions**2)))
        b_bias[i-1]= (predictions - targets).mean()
        ## Calculate standard deviation of 'b' for these points
        b_std[i - 1] = np.std(predictions - b_bias[i-1])
        MFE[i-1]=np.median(10**(np.abs(np.log10(predictions/targets))-1))
        #rel_100=
        MRE[i-1]=np.mean(np.abs(predictions - targets)/predictions)*100
    my_RMSE = b_error/bin_centers
    
    
    # Plotting
    fig, axs = plt.subplots(figsize=(5,6))
    
    # Plot Error curves
    
    x=np.logspace(bin_min, bin_max, num_bins *10)
    if variable == 'IWV':
        error_curves =[1,5,10,25]
    else:
        error_curves =[5,10,25,50,100]
    
    for e in error_curves:
        #plt.plot(x,-(0.01*e)*x,linestyle=':',color='grey',label=e)
        #plt.plot(x,(0.01*e)*x,linestyle=':',color='grey',label=e) #relative erorr 20 percent
        X=1200#1200#400 + (0.01*e)*800
        #plt.text(X*(1/(0.01*e)), X,f'{e} %',verticalalignment='center_baseline',horizontalalignment='center',rotation=80)
        #plt.scatter(X*(1/(0.01*e)), X)
    #plt.plot(x,0*x,linestyle='-',color='grey',label=e)
    plt.plot(bin_centers,100*my_RMSE,label='RMSE')  
    #plt.plot(bin_centers,100*RRMSE,label='RMSE')  
    plt.plot(bin_centers,MRE,label='MRE')
    #plt.plot(bin_centers,MFE,label='MFE')
    #plt.plot(bin_centers,b_error)
    #plt.plot(bin_centers,b_bias)
    #plt.errorbar(bin_centers, np.zeros_like(bin_centers), yerr=b_error, fmt='o', capsize=5)
    plt.legend()
    if variable == 'IWV':
        plt.xlabel('Retrieved '+ variable+'[kg m$^{-2}$]')
        plt.plot(bin_centers,0.05*bin_centers,linestyle='--',color='grey') #relative erorr 100 percent
        plt.plot(bin_centers,0.01*bin_centers,linestyle='-.',color='grey') #relative erorr 100 percent
    else:
        plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('Retrieved '+ variable+' \\ g m$^{-2}$')
        
        #if  variable == 'LWP':
            
    plt.ylim(bottom=0,top=300)
        #elif  variable == 'LWP':
            #plt.ylim(bottom=-150,top=1300)
            #plt.yscale('log')
        #else:
            #plt.ylim(top=120)       

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.ylabel('Root mean square error \\ percent')
    #plt.title('Error of b within each a bin')
    plt.tight_layout()
    if filename != None:
        plt.savefig(f'/home/u/u301032/orcestra/plots/{variable}_error_NN_{filename}.png',dpi=400)
    plt.show()
    print(bin_centers,b_bias)
