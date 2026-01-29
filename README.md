# NN_IWP_retrieval
This repository contains all necassary files of the neural-network-based IWV, LWP and IWP retrieval for the HALO-PERCUSION observations. In general, the retrieval consists of identical neural networks having the following architecture:

<img width="594" alt="Screenshot 2023-11-07 at 18 17 49" src="https://github.com/MaxRing96/NN_IWP_retrieval/assets/62293752/a9208f42-d2eb-40bc-ac96-d002d60f22cc">

For each of 8 altitude levels between 11 km and 15 km, three nerual networks have been trained separatley for IWP LWP and IWV.
The training/testing dataset was created by a coupled model set-up of ICON and PAMTRA. The final retrieval outputs closest to the altitude level closest to the actual HAMP observation. Detailed information about the retrieval and its development can be found in the master's thesis document.

All the trained neural networks itself are saved under the [NNs](NNs) directory.

### For training and development of the Neural Networs these Notebooks have been used:

- Subselection of hydrometeors compared to all hydrometeors in [NNs/hydrometeor-climatology.ipynb](NNs/hydrometeor-climatology.ipynb)
- Saving training and testing data as numpy arrays for keras [NN/prep-for-NN-training.py](NN/prep-for-NN-training.py)
- Development and Training of NNs [NNs/NN_development.ipynb](NNs/NN_development.ipynb) with source code in [NNs/src.py](NNs/src.py)

### The analysis of the retrieved data is in:
- [retrieval-product-analysis.ipynb](retrieval-product-analysis.ipynb)

### The analysis of hydrometeors to cloud level and IWV is in:
- [cloud-layers.ipynb](cloud-layers.ipynb) with additional plots in [Additional_plots.ipynb](Additional_plots.ipynb)



However, the radar and radiometer data is not included in this repository and has to be downloaded seperately.

## How to apply the retrieval?
To apply the retrieval on the HALO-PERCUSION HAMP radiometer observations, you only need to download this repository (the notebook is not needed) along with the following python packages:
  - numpy
  - xarray
  - tensorflow
  - glob

as well as the HAMP radiometer data.






