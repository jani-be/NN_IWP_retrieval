# NN_IWP_retrieval
Neural-network-based ice water path retrieval for the passive airborne microwave observations of HAMP, conducted during the HALO-AC3 campaign

This repository contains all necassary files of the neural-network-based IWP retrieval for the HALO-AC3 observations. In general, the retrieval consists of 30 identical neural networks having the following architecture:
<img width="594" alt="Screenshot 2023-11-07 at 18 17 49" src="https://github.com/MaxRing96/NN_IWP_retrieval/assets/62293752/a9208f42-d2eb-40bc-ac96-d002d60f22cc">


For each of 10 altitude levels between 8.5 km and 13 km, three nerual networks have been trained on different random subsamples of the same training/testing dataset.
Last one was created by a coupled model set-up of ICON and PAMTRA. The final retrieval outputs an average of the three networks trained at the altitude level closest to the actual HAMP observation. Detailed information about the retrieval and its development can be found in the master's thesis document.

All the trained neural networks itself are saved under the [NNs](NNs) directory, which inlcudes all individual networks trained for altitudes between 8.5 km and 13 km.
Along with these networks the (ICON-PAMTRA) training and testing data used for each of the networks is saved as netcdf files under [train_test_data](NNs\train_test_data).
Further, figures displaying the [loss](NNs\testing_loss) during training as well as the network predictions for each of their test data set (as [probability density functions](NNs\testing_pdfs) and [scatterdensity](NNs\testing_scatterdensity) plots) are saved.

The python script [src.py](src.py) contains some basic functions used in the other scripts, while [nn_iwp_retrieval.py](nn_iwp_retrieval.py) contains the core function of the neural network retrieval. 
The jupyter notebook [retrieval_quicklooks.ipynb](retrieval_quicklooks.ipynb) can be used to apply the retrieval on the individual HALO-AC3 research flights and to plot its results along with the HAMP radar and radiometer data.
However, the radar and radiometer data is not included in this repository and has to be downloaded seperately in order to use the notebook. The notebook just serves as an example for the retrieval application, but is not needed for the retrieval application in general

## How to apply the retrieval?
To apply the retrieval on the HALO-AC3 HAMP radiometer observations, you only need to download this repository (the notebook is not needed) along with the following python packages:
  - numpy
  - xarray
  - tensorflow
  - glob

as well as the HAMP radiometer data (no public server so far).

To retrieve the IWP from the HAMP radiometer (=brightness temperature) observations you have to `from nn_iwp_retrieval import retrieve_IWP` and use the retrieval function as `retrieve_IWP(HAMP_TBs,HAMP_alt)`.
Further detailed information, e.g. how the two arrays of HAMP brightness temperatures (HAMP_TBs) and observation altitudes (HAMP_alt) have to be arranged, can be found within the [nn_iwp_retrieval.py](nn_iwp_retrieval.py) script.

If you have questions or comments on the retrieval you can contact me via: maximilian_ringel@icloud.com.

