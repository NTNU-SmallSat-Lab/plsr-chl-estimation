#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path

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
from scipy import ndimage
from skimage.morphology import disk




sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso')
sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso2_calibration')

#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso1_calibration')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso2_calibration')

from hypso import Hypso

# Path to the base directory
#base_dir = "/home/_shared/ARIEL/PLSR/captures"
#datasets_dir = "/home/_shared/ARIEL/PLSR/datasets"

base_dir = "/home/camerop/ARIEL/PLSR/captures_ocx"
combined_datasets_dir = "/home/_shared/ARIEL/PLSR/datasets_ocx"
h1_datasets_dir = "/home/_shared/ARIEL/PLSR/datasets_h1_ocx"
h2_datasets_dir = "/home/_shared/ARIEL/PLSR/datasets_h2_ocx"


os.makedirs(base_dir, exist_ok=True)
os.makedirs(combined_datasets_dir, exist_ok=True)
os.makedirs(h1_datasets_dir, exist_ok=True)
os.makedirs(h2_datasets_dir, exist_ok=True)

script_dir = os.path.dirname(os.path.realpath(__file__))


# Iterate over all entries in the base directory
for entry in os.listdir(base_dir):
    full_path = os.path.join(base_dir, entry)
    
    # Check if the entry is a directory
    #if os.path.isdir(full_path):

    folder_name = os.path.basename(full_path)
    l1d_nc_path = os.path.join(full_path, f"{folder_name}-l1d.nc")
    slc_nc_path = os.path.join(full_path, f"{folder_name}-slc.nc")
    l2_6s_path = os.path.join(full_path, f"{folder_name}-l2a-6sv1.nc")

    pattern = os.path.join(full_path, f"{folder_name}_sentinel_chl*.nc")
    sentinel_nc_paths = glob.glob(pattern)

    # Load the data

    try:

        ## Load the HYPSO data
        #satobj = Hypso(path=l1d_nc_path, verbose=True)
        #hypso_data = satobj.l1d_cube.to_numpy()

        satobj = Hypso(path=l2_6s_path, verbose=True)
        hypso_data = satobj.l2a_cubes["6sv1"].to_numpy()



        sensor = satobj.sensor

        match sensor:
            case "hypso1_hsi":
                datasets_dir = h1_datasets_dir
            case "hypso2_hsi":
                datasets_dir = h2_datasets_dir
            case _:
                datasets_dir = combined_datasets_dir



        # Load the HYPSO mask
        with Dataset(slc_nc_path, "r") as ncfile:
            # Read dimensions
            y_dim = ncfile.dimensions["y"].size
            x_dim = ncfile.dimensions["x"].size

            # Read variables
            lats = ncfile.variables["lat"][:, :]
            lons = ncfile.variables["lon"][:, :]
            
            hypso_land_mask = ncfile.variables["land"][:, :].astype(bool)
            hypso_cloud_mask = ncfile.variables["cloud"][:, :].astype(bool)
            
            #hypso_mask = ncfile.variables["water"][:, :]

            hypso_mask = hypso_land_mask | hypso_cloud_mask



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

            mask = sentinel_mask.astype(bool) | hypso_mask.astype(bool)


            footprint = disk(16) # pixel extent enlargment
            mask = ndimage.binary_dilation(mask, structure=footprint)


            plt.imshow(mask)
            plt.savefig(os.path.join(datasets_dir, satobj.capture_name + '_dialated_mask.png'))
            plt.close()



            X = np.where(~mask[:, :, np.newaxis], hypso_data, np.nan)
            Y = np.where(~mask, sentinel_chl, np.nan)


            #X = X[:, :,6:-6]

            #X = X[~mask][:, :,6:-6]
            #Y = Y[~mask]

            #X = np.clip(X, 0, 1)

            #Y = 10**Y
            #Y = np.clip(Y, 0, 30)
            

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
            





