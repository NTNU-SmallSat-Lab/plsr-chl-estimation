#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run PLSR for IGARSS Chl-a PLSR estimation

Author: Cameron Penne
Date: 2025-01-06
"""

import sys
sys.path.insert(0, '/home/cameron/Projects/hypso-package')

from hypso import Hypso1
import os
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import csv
from pyresample import load_area
import glob
from satpy import Scene
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler
from pyresample.bilinear.xarr import XArrayBilinearResampler 
import pickle
import glob
import pickle

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

import os
import re
from collections import defaultdict

script_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "dataset")


l1a_nc_file = "/home/cameron/Dokumenter/129.241.2.147:8008/roervik/roervik_2024-05-05T10-39-17Z/roervik_2024-05-05T10-39-17Z-l1a.nc"
points_file = "/home/cameron/Projects/hypso1-qgis-gcps/png/bin3/roervik/roervik_2024-05-05_1039Z-bin3.points"


satobj = Hypso1(path=l1a_nc_file, verbose=True)
satobj.load_points_file(path=points_file, image_mode='standard', origin_mode='cube')

satobj.generate_l1b_cube()
satobj.generate_l1c_cube()
satobj.generate_l1d_cube()

def decode_labels(file_path):
    # Open the binary file and read its content
    with open(file_path, 'rb') as fileID:
        fileContent = fileID.read()
    # Extract the required values from the binary data
    classification_execution_time = int.from_bytes(fileContent[0:4], byteorder='little', signed=True)
    loading_execution_time = int.from_bytes(fileContent[4:8], byteorder='little', signed=True)
    classes_holder = fileContent[8:24]
    labels_holder = fileContent[24:]
    classes = []
    labels = []
    # Decode the labels and convert them back to original classes.
    for i in range(len(classes_holder)):
        if classes_holder[i] != 255:
            classes.append(classes_holder[i])
    if len(classes) <= 2:
        for i in range(len(labels_holder)):
            pixel_str = format(labels_holder[i], '08b')
            for j in range(8):
                labels.append(int(pixel_str[j]))
    if 2 < len(classes) <= 4:
        for i in range(len(labels_holder)):
            pixel_str = format(labels_holder[i], '08b')
            for j in range(4):
                labels.append(int(pixel_str[2 * j:2 * j + 2], 2))
    if 4 < len(classes) <= 16:
        for i in range(len(labels_holder)):
            pixel_str = format(labels_holder[i], '08b')
            for j in range(2):
                labels.append(int(pixel_str[4 * j:4 * j + 4], 2))
    # Corrected label conversion
    for i in range(len(labels)):
        labels[i] = classes[labels[i]]
    # Save 'labels' as a CSV file with a comma delimiter
    # with open('labels.csv', 'w') as csv_file:
    #    csv_file.write(','.join(map(str, labels)))
    return labels

labels = {'water': 0,
        'strange_water': 1,
        'light_forest': 2,
        'dark_forest': 3,
        'urban': 4,
        'rock': 5,
        'ice': 6,
        'sand': 7,
        'thick_clouds': 8,
        'thin_clouds': 9,
        'shadows': 10}


labels_data = decode_labels(labels_path)
lsc = np.array(labels_data).reshape(satobj.spatial_dimensions)

thick_cloud_key = labels['thick_clouds']
thin_cloud_key = labels['thin_clouds']
water_key = labels['water']
ice_key = labels['ice']

cloud_mask = ((lsc == thick_cloud_key) | (lsc == thin_cloud_key))
land_mask = ~(lsc == water_key)

satobj.cloud_mask = cloud_mask
satobj.land_mask = land_mask

satobj.generate_toa_reflectance()
#satobj.generate_l2a_cube('machi')









scene = satobj.get_toa_reflectance_satpy_scene()
hypso_swath_def = scene['band_1'].attrs['area']

filenames = []
filenames = filenames + glob.glob(sentinel_path + '/geo_coordinates.nc')
filenames = filenames + glob.glob(sentinel_path + '/chl_nn.nc')
sentinel_scene = Scene(filenames=filenames, reader='olci_l2')
sentinel_scene.load(['chl_nn'])

sentinel_swath_def = sentinel_scene['chl_nn'].attrs['area']
nnrs = KDTreeNearestXarrayResampler(source_geo_def=sentinel_swath_def, target_geo_def=hypso_swath_def)
sentinel_chl = nnrs.resample(sentinel_scene['chl_nn'], fill_value=np.nan, radius_of_influence=None, epsilon=0)


X = satobj.toa_reflectance_cube[:,:,6:]
#X = satobj.l2a_cube[:,:,6:].fillna(0)
X_dims = X.shape
X = X.to_numpy().reshape(-1,114)




components = [4, 8, 16, 32]
transform = ['linear', 'log']
results = {
  "linear" : {},
  "log" : {}
}

sentinel_mask = np.isnan(sentinel_chl) #| np.isnan(hypso_chl)
hypso_mask = land_mask | cloud_mask

mask = sentinel_mask | hypso_mask


for t in transform:

    for c in components:

        print("Running " + t + " with " + str(c) + " components.")

        pls_model_path = os.path.join(output_dir, "pls_model_" + t + "_" + str(c) + ".pkl")

        with open(pls_model_path, 'rb') as file:
            pls = pickle.load(file)

        Y = pls.predict(X)
        Y = Y.reshape(X_dims[0], X_dims[1], -1)
        Y = Y[:,:,0]
        
        if t == 'linear':
            # Convert sentinel to linear values
            sentinel_chl_scoring = sentinel_chl
            sentinel_chl_scoring = np.where(~mask, sentinel_chl_scoring, np.nan)
            sentinel_chl_scoring = np.clip(sentinel_chl_scoring, 0, None)
            
            # Already linear
            hypso_chl_scoring = Y 
            hypso_chl_scoring = np.where(~mask, hypso_chl_scoring, np.nan)
            hypso_chl_scoring = np.clip(hypso_chl_scoring, 0, None)
            hypso_chl_plotting = hypso_chl_scoring

        elif t == 'log':
            # Already log
            sentinel_chl_scoring = np.log10(sentinel_chl)
            sentinel_chl_scoring = np.where(~mask, sentinel_chl_scoring, np.nan)

            # Already log
            hypso_chl_scoring = Y 
            hypso_chl_scoring = np.where(~mask, hypso_chl_scoring, np.nan)
            hypso_chl_plotting = 10**hypso_chl_scoring
        else:
            break

        #if t == 'log':
        #    hypso_chl_scoring = 10**hypso_chl
        #    hypso_chl_scoring = hypso_chl
        #    hypso_chl_scoring = np.where(~mask, hypso_chl_scoring, np.nan)
        #    hypso_chl_scoring = np.clip(hypso_chl_scoring, 0, None)

        nan_mask = np.isnan(sentinel_chl_scoring) | np.isnan(hypso_chl_scoring)
        inf_mask = np.isinf(sentinel_chl_scoring) | np.isinf(hypso_chl_scoring)

        filter_mask = nan_mask | inf_mask

        hypso_chl_scoring_filtered = hypso_chl_scoring[~filter_mask]
        sentinel_chl_scoring_filtered = sentinel_chl_scoring[~filter_mask]

        rmse = root_mean_squared_error(sentinel_chl_scoring_filtered, hypso_chl_scoring_filtered)
        r2 = r2_score(sentinel_chl_scoring_filtered, hypso_chl_scoring_filtered)

        print(rmse)
        print(r2)


        plt.figure(figsize=(8, 5))
        #plt.imshow(np.rot90(hypso_chl_scoring[:, ::-3], k=3))
        plt.imshow(np.rot90(hypso_chl_plotting[:, ::-3], k=3), vmin=0, vmax=15)
        cbar = plt.colorbar(fraction=0.03, pad=0.04, extend='max')
        cbar.ax.set_ylabel('Chl-a concentration [mg/m^3]')
        plt.tight_layout()
        plt.savefig('roervik_' + t + str(c) + '_hypso.png')
        plt.show()
















'''
plt.figure(figsize=(8, 5))
img = mpimg.imread('/home/cameron/Nedlastinger/roervik_2024-05-05T10-39-17Z-bin3.png')
plt.imshow(np.rot90(img, k=3))
cbar = plt.colorbar(fraction=0.03, pad=0.04, extend='max')
cbar.ax.set_ylabel('Chl-a concentration [mg/m^3]')
plt.tight_layout()
plt.savefig('roervik_rgb.png')
plt.show()

plt.figure(figsize=(8, 5))
plt.imshow(np.rot90(hypso_chl_scoring[:, ::-3], k=3), vmin=0, vmax=15)
cbar = plt.colorbar(fraction=0.03, pad=0.04, extend='max')
cbar.ax.set_ylabel('Chl-a concentration [mg/m^3]')
plt.tight_layout()
plt.savefig('roervik_hypso.png')
plt.show()

plt.figure(figsize=(8, 5))
plt.imshow(np.rot90(sentinel_chl_scoring[:, ::-3], k=3), vmin=0, vmax=15)
cbar = plt.colorbar(fraction=0.03, pad=0.04, extend='max')
cbar.ax.set_ylabel('Chl-a concentration [mg/m^3]')
plt.tight_layout()
plt.savefig('roervik_sentinel.png')
plt.show()
'''















