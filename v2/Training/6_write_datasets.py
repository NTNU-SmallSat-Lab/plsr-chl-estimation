#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path
from hypso import Hypso
import glob
from satpy import Scene
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler
from pyresample.bilinear.xarr import XArrayBilinearResampler 
from pyresample.geometry import SwathDefinition, AreaDefinition
import xarray as xr
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import subprocess
import pickle



# Path to the base directory
base_dir = "/home/_shared/ARIEL/PLSR/captures"
datasets_dir = "/home/_shared/ARIEL/PLSR/datasets"

os.makedirs(base_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)

script_dir = os.path.dirname(os.path.realpath(__file__))


# Iterate over all entries in the base directory
for entry in os.listdir(base_dir):
    full_path = os.path.join(base_dir, entry)
    
    # Check if the entry is a directory
    #if os.path.isdir(full_path):

    folder_name = os.path.basename(full_path)
    l1d_nc_path = os.path.join(full_path, f"{folder_name}-l1d.nc")
    slc_nc_path = os.path.join(full_path, f"{folder_name}-slc.nc")

    pattern = os.path.join(full_path, f"{folder_name}_sentinel_chl_*.nc")
    sentinel_nc_paths = glob.glob(pattern)

    # Load the data

    try:

        ## Load the HYPSO data
        satobj = Hypso(path=l1d_nc_path, verbose=True)
        hypso_data = satobj.l1d_cube.to_numpy()

        # Load the HYPSO mask
        with Dataset(slc_nc_path, "r") as ncfile:
            # Read dimensions
            y_dim = ncfile.dimensions["y"].size
            x_dim = ncfile.dimensions["x"].size

            # Read variables
            lats = ncfile.variables["lat"][:, :]
            lons = ncfile.variables["lon"][:, :]
            hypso_mask = ncfile.variables["water"][:, :]

        # Load the Sentinel data and mask
        for i, sentinel_nc_path in enumerate(sentinel_nc_paths):

            with Dataset(sentinel_nc_path, "r") as ncfile:
                # Read dimensions
                y_dim = ncfile.dimensions["y"].size
                x_dim = ncfile.dimensions["x"].size

                # Read variables
                lats = ncfile.variables["lat"][:, :]
                lons = ncfile.variables["lon"][:, :]
                sentinel_chl = ncfile.variables["chl_nn"][:, :]
                sentinel_mask = ncfile.variables["mask"][:, :]

            mask = sentinel_mask.astype(bool) | ~hypso_mask.astype(bool)

            X = np.where(~mask[:, :, np.newaxis], hypso_data, np.nan)
            Y = np.where(~mask, sentinel_chl, np.nan)



        
            X = X[:, :,6:-6]
            Y = Y

            #X = X[~mask][:, :,6:-6]
            #Y = Y[~mask]

            X = np.clip(X, 0, 1)

            Y = 10**Y
            Y = np.clip(Y, 0, 10)
            

            plt.imshow(X[:,:,40])
            plt.savefig(os.path.join(datasets_dir, satobj.capture_name + '_band_40_' + str(i) + '.png'))
            plt.close()

            plt.imshow(Y)
            plt.savefig(os.path.join(datasets_dir, satobj.capture_name + '_sentienl_chl_' + str(i) + '.png'))
            plt.close()

            X = X[~mask]
            Y = Y[~mask]

            print(X.shape)
            print(Y.shape)

            dataset_path = os.path.join(datasets_dir, satobj.capture_name + '_dataset_' + str(i) + '.pkl')

            dataset = {
                'X': X,
                'Y': Y
            }

            with open(dataset_path, 'wb') as file:
                pickle.dump(dataset, file)

    except Exception as ex:
        print(ex)
        continue
            





