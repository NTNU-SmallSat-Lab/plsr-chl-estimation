#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create dataset for IGARSS Chl-a PLSR estimation

Author: Cameron Penne
Date: 2024-02-27
"""

import sys
sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso/')

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

from hypso.classification import decode_jon_cnn_water_mask, decode_jon_cnn_land_mask, decode_jon_cnn_cloud_mask

from hypso.geometry_definition import generate_hypso_swath_def

def read_csv(file_path):
    data_list = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            if len(row) >= 3:
                entry_dict = {}
                entry_dict['enable'] = row[0].lower() == 'true'
                entry_dict['hypso'] = row[1]
                entry_dict['sentinel'] = row[2]
                entry_dict['points'] = row[3]
                entry_dict['lats'] = row[4]
                entry_dict['lons'] = row[5]
                entry_dict['flip'] = row[6].lower() == 'true'

                data_list.append(entry_dict)

    return data_list




sensor = "h2" # h1 or h2

captures_csv_path = './captures_' + sensor + '.csv'

entry_list = read_csv(captures_csv_path)
points_file_base_dir = "/home/cameron/Projects/hypso1-qgis-gcps/png/bin3"

script_dir = os.path.dirname(os.path.abspath(__file__))


output_dir = os.path.join(script_dir, "dataset_" + sensor)

for entry in entry_list:

    print('Processing capture:')
    print(entry['hypso'])

    if not entry['enable']:
        print('Capture not enabled. Skipping.')
        continue

    hypso_path = entry['hypso']
    sentinel_path = entry['sentinel']

    parts = hypso_path.rstrip('/').split('/')

    print(parts)

    capture_target_name = parts[-2]

    print(capture_target_name)

    timestamp = parts[-1].split('_')[1]

    base_labels_path = os.path.join(hypso_path, "processing-temp")

    print(base_labels_path)

    for item in os.listdir(base_labels_path):
        if os.path.isdir(os.path.join(base_labels_path, item)) and item.startswith(capture_target_name):
            subdir_name = item
            break

    if sensor == 'h1':
        labels_path = os.path.join(base_labels_path, "jon-cnn.labels")
    else:
        labels_path = os.path.join(base_labels_path, "sea-land-cloud.labels")





    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H-%M-%SZ")
    new_timestamp = dt.strftime("%Y-%m-%d_%H%MZ")

    points_file = os.path.join(points_file_base_dir, capture_target_name, f"{capture_target_name}_{new_timestamp}-bin3.points")


    name = f"{capture_target_name}_{timestamp}-l1a"
    l1a_nc_file = os.path.join(hypso_path, name + '.nc')

    print("Hypso Path: ", hypso_path)
    print("Labels Path: ", labels_path)
    print("Points File: ", points_file)
    print("Sentinel Path: ", sentinel_path)
    
    satobj = Hypso1(path=l1a_nc_file, verbose=True)

    land_mask = decode_jon_cnn_land_mask(file_path=labels_path, spatial_dimensions=satobj.spatial_dimensions)
    cloud_mask = decode_jon_cnn_cloud_mask(file_path=labels_path, spatial_dimensions=satobj.spatial_dimensions)
    water_mask = decode_jon_cnn_water_mask(file_path=labels_path, spatial_dimensions=satobj.spatial_dimensions)

    satobj.land_mask = land_mask
    satobj.cloud_mask = cloud_mask
    
    mask = satobj._unified_mask()

    plt.imshow(land_mask, interpolation='nearest')
    plt.savefig('land_mask.png')

    plt.imshow(cloud_mask, interpolation='nearest')
    plt.savefig('cloud_mask.png')

    plt.imshow(water_mask, interpolation='nearest')
    plt.savefig('water_mask.png')

    plt.imshow(satobj._unified_mask(), interpolation='nearest')
    plt.savefig('mask.png')

    plt.imshow(satobj.l1a_cube[:,:,40], interpolation='nearest')
    plt.savefig('l1a_greyscale.png')

    #input("Pause...")

    satobj.generate_l1b_cube()
    satobj.generate_l1c_cube()
    satobj.generate_l1d_cube()

    plt.imshow(satobj.l1d_cube[:,:,40], interpolation='nearest')
    plt.savefig('l1d_greyscale.png')

    #input("Pause...")


    if entry['lats'] and entry['lons']:

        # Read from latitudes_indirectgeoref.dat
        with open(entry['lats'], mode='rb') as file:
            file_content = file.read()
        
        lats = np.frombuffer(file_content, dtype=np.float32)

        lats = lats.reshape(satobj.spatial_dimensions)

        # Read from longitudes_indirectgeoref.dat
        with open(entry['lons'], mode='rb') as file:
            file_content = file.read()
        
        lons = np.frombuffer(file_content, dtype=np.float32)

        lons = lons.reshape(satobj.spatial_dimensions)

        # Directly provide the indirect lat/lons loaded from the file. This function will run the track geometry computations.
        satobj.run_indirect_georeferencing(latitudes=lats, longitudes=lons)

    else:
        satobj.run_indirect_georeferencing(points_file_path=points_file)

    swath_def = generate_hypso_swath_def(satobj, use_indirect_georef=True)

    plt.imshow(satobj.latitudes_indirect, interpolation='nearest')
    plt.savefig('latitudes_indirect.png')

    plt.imshow(satobj.longitudes_indirect, interpolation='nearest')
    plt.savefig('longitudes_indirect.png')

    plt.imshow(satobj.latitudes, interpolation='nearest')
    plt.savefig('latitudes.png')

    plt.imshow(satobj.longitudes, interpolation='nearest')
    plt.savefig('longitudes.png')


    #input("Pause...")

    filenames = []
    filenames = filenames + glob.glob(sentinel_path + '/geo_coordinates.nc')
    filenames = filenames + glob.glob(sentinel_path + '/chl_nn.nc')
    filenames = filenames + glob.glob(sentinel_path + '/wqsf.nc')
    sentinel_scene = Scene(filenames=filenames, reader='olci_l2')
    sentinel_scene.load(['chl_nn', 'mask'])

    #input("Pause...")


    resampled_sentinel_scene = sentinel_scene.resample(swath_def, resampler='nearest', fill_value=np.nan)
    #resampled_sentinel = resampled_sentinel_scene.to_xarray()
    resampled_sentinel = resampled_sentinel_scene['chl_nn']


    resampled_sentinel = np.where(satobj._unified_mask(), np.nan, resampled_sentinel)

    plt.close()
    plt.imshow(resampled_sentinel, interpolation='nearest')
    plt.savefig('chl_nn.png')

    #input("Pause...")


    basename = satobj.capture_name

    proc_mask_path = os.path.join(output_dir, basename + "_mask.pkl")
    proc_hypso_path = os.path.join(output_dir, basename + "_hypso.pkl")
    proc_sentinel_path = os.path.join(output_dir, basename + "_sentinel.pkl")


    with open(proc_mask_path, 'wb') as file:
        pickle.dump(mask, file)

    with open(proc_hypso_path, 'wb') as file:
        pickle.dump(satobj.l1d_cube.to_numpy(), file)

    with open(proc_sentinel_path, 'wb') as file:
        pickle.dump(resampled_sentinel, file)

    input("Press Enter to continue to next capture...")
