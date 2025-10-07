#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run PLSR for IGARSS Chl-a PLSR estimation

Author: Cameron Penne
Date: 2025-01-06
"""

#import sys
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso/')

from hypso import Hypso
import os
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from pyresample import load_area
import pickle
from sklearn.cross_decomposition import PLSRegression
import os
import h5py

components = 10

script_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))


datasets_dir = "/home/_shared/ARIEL/PLSR/datasets"
dataset_file = os.path.join(datasets_dir, "combined_dataset.h5")
model_file = os.path.join(datasets_dir, "pls_model_c" + str(components) + ".h5")

# Open the HDF5 file in read mode
with h5py.File(dataset_file, 'r') as h5f:
    # Access datasets
    X = h5f['X'][:]
    Y = h5f['Y'][:]

    # Print shapes
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")


print(X.shape)
print(Y.shape)




print("Running with " + str(components) + " components.")

pls = PLSRegression(n_components=components, max_iter=500)
##scoring = ['explained_variance', 'r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error']
##cv = KFold(n_splits=10, shuffle=True)
##scores = cross_validate(pls, X, Y, cv=cv, scoring=scoring, return_indices=True)

##print(scores)

pls.fit(X,Y)
pls_model_path = os.path.join(model_file, "pls_model_c" + str(components) + ".pkl")
with open(pls_model_path, 'wb') as file:
    pickle.dump(pls, file)



