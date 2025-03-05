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

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

import os
import re
from collections import defaultdict

script_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "dataset")

dataset_X_path = os.path.join(output_dir, "dataset_X.pkl")
dataset_Y_path = os.path.join(output_dir, "dataset_Y.pkl")


with open(dataset_X_path, 'rb') as file:
    X_loaded = pickle.load(file)

with open(dataset_Y_path, 'rb') as file:
    Y_loaded = pickle.load(file)

components = 10

X = X_loaded
Y = Y_loaded # chl_nn is log values, reverse
#Y = 10**Y_loaded # chl_nn is log values, reverse
#Y = np.clip(Y, 0, None)

        

print("Running with " + str(components) + " components.")

pls = PLSRegression(n_components=components, max_iter=500)
#scoring = ['explained_variance', 'r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error']
#cv = KFold(n_splits=10, shuffle=True)
#scores = cross_validate(pls, X, Y, cv=cv, scoring=scoring, return_indices=True)

#print(scores)

pls.fit(X,Y)
pls_model_path = os.path.join(output_dir, "pls_model_c" + str(components) + ".pkl")
with open(pls_model_path, 'wb') as file:
    pickle.dump(pls, file)


