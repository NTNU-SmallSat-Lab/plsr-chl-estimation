#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load dataset for IGARSS Chl-a PLSR estimation

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
from datetime import datetime
import csv
from pyresample import load_area
import glob
from satpy import Scene
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler
from pyresample.bilinear.xarr import XArrayBilinearResampler 
import pickle
import glob

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import os
import re
from collections import defaultdict

script_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "dataset")

# Dictionary to store matched files
matched_files = defaultdict(list)

# Regular expression to extract the prefix and timestamp
pattern = re.compile(r"^(.*?_\\d{4}-\\d{2}-\\d{2}T\\d{2}-\\d{2}-\\d{2}Z)_.*\\.pkl$")
pattern = re.compile("^(.*?_\\d{4}-\\d{2}-\\d{2}T\\d{2}-\\d{2}-\\d{2}Z)_.*\\.pkl$")

# Iterate over each file in the directory and group them by prefix and timestamp
for file in os.listdir(output_dir):
    if file.endswith('.pkl'):
        match = pattern.match(file)
        if match:
            key = match.group(1)
            matched_files[key].append(file)

for key in matched_files:
    matched_files[key].sort()

# Print the matched files
for key, group in matched_files.items():
    print(f"{key}: {group}")

input("Press any key to begin loading...")


X_matrices = []
Y_matrices = []

for key, group in matched_files.items():

    name = key

    hypso_path = os.path.join(output_dir, group[0])
    mask_path = os.path.join(output_dir, group[1])
    sentinel_path = os.path.join(output_dir, group[2])

    print(name)


    with open(hypso_path, 'rb') as file:
        hypso_data = pickle.load(file)

    with open(mask_path, 'rb') as file:
        mask_data = pickle.load(file)

    with open(sentinel_path, 'rb') as file:
        sentinel_data = pickle.load(file)


    plt.imshow(hypso_data[:,:,100])
    plt.savefig(os.path.join(output_dir, name + "_1_hypso.png"))
    plt.clf()

    plt.imshow(mask_data)
    plt.savefig(os.path.join(output_dir, name + "_2_mask.png"))
    plt.clf()

    plt.imshow(sentinel_data)
    plt.savefig(os.path.join(output_dir, name + "_3_sentinel.png"))
    plt.clf()

    hypso_mask = mask_data.to_numpy()
    sentinel_mask = np.isnan(sentinel_data)

    mask = hypso_mask | sentinel_mask

    X = np.where(~mask[:, :, np.newaxis], hypso_data, np.nan)
    Y = np.where(~mask, sentinel_data, np.nan)

    print('min/max values before:')
    print(np.nanmin(X))
    print(np.nanmax(X))

    X = X[~mask][:,6:]
    Y = Y[~mask]

    # Move to loading?
    nan_indices = np.where(np.isnan(X).any(axis=1))[0]
    X = np.delete(X, nan_indices, axis=0)
    Y = np.delete(Y, nan_indices, axis=0)

    #X = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))
    #X = np.clip(X, 0, 1)

    print('min/max values after:')
    print(np.nanmin(X))
    print(np.nanmax(X))


    print('X & Y shapes:')
    print(X.shape)
    print(Y.shape)

    X_matrices.append(X)
    Y_matrices.append(Y)

del X, Y
X = np.vstack(X_matrices)
del X_matrices
Y = np.concatenate(Y_matrices)
del Y_matrices

print(X.shape)
print(Y.shape)


dataset_X_path = os.path.join(output_dir, "dataset_X.pkl")
dataset_Y_path = os.path.join(output_dir, "dataset_Y.pkl")

with open(dataset_X_path, 'wb') as file:
    pickle.dump(X, file)

with open(dataset_Y_path, 'wb') as file:
    pickle.dump(Y, file)