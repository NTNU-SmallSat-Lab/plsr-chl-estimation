#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run PLSR for IGARSS Chl-a PLSR estimation

Author: Cameron Penne
Date: 2025-10-07
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

components = 10

script_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))


datasets_dir = "/home/_shared/ARIEL/PLSR/datasets"
dataset_file = os.path.join(datasets_dir, "combined_dataset.h5")
#model_file = os.path.join(datasets_dir, "pls_model_c" + str(components) + ".h5")

# Open the HDF5 file in read mode
with h5py.File(dataset_file, 'r') as h5f:
    # Access datasets
    X = h5f['X'][:]
    Y = h5f['Y'][:]

    # Print shapes
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

for components in [4, 8, 16, 32]:

    print("Running PLS with " + str(components) + " components.")

    pls = PLSRegression(n_components=components, max_iter=500)
    scoring = ['explained_variance', 'r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error']
    cv = KFold(n_splits=5, shuffle=True)
    scores = cross_validate(pls, X, Y, cv=cv, scoring=scoring, return_indices=True, verbose=1)

    for key in scores.keys():
        print(key)
        print(scores[key])


    # Write scores to a text file
    score_file_path = os.path.join(datasets_dir, f"pls_scores_c{components}.txt")
    with open(score_file_path, 'w') as f:
        f.write(f"Scores for PLS with {components} components:\n\n")
        for key in scores:
            f.write(f"{key}:\n{scores[key]}\n\n")



    pls.fit(X,Y)
    pls_model_path = os.path.join(datasets_dir, "pls_model_c" + str(components) + ".pkl")

    with open(pls_model_path, 'wb') as file:
        pickle.dump(pls, file)

print("Done!")



