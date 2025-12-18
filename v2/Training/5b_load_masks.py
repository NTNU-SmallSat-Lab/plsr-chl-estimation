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


sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso')
sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso2_calibration')

#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso1_calibration')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso2_calibration')

from hypso import Hypso

def main(l1d_nc_path, lats_path=None, lons_path=None):
    # Check if the first file exists
    if not os.path.isfile(l1d_nc_path):
        print(f"Error: The file '{l1d_nc_path}' does not exist.")
        return

    # Process the first file
    print(f"Processing file: {l1d_nc_path}")

    nc_file = Path(l1d_nc_path)

    satobj = Hypso(path=nc_file, verbose=True)

    lats = satobj.latitudes
    lons = satobj.longitudes

    #lats = satobj.latitudes_indirect
    #lons = satobj.longitudes_indirect

    spatial_dimensions = satobj.spatial_dimensions
    
    full_path = os.path.join(Path(l1d_nc_path).parent, "processing-temp")


    labels_path = os.path.join(full_path, "sea-land-cloud.labels")

    from hypso.classification import decode_jon_cnn_labels, decode_jon_cnn_cloud_mask, decode_jon_cnn_water_mask, decode_jon_cnn_land_mask

    labels_arr = decode_jon_cnn_labels(file_path=labels_path, spatial_dimensions=spatial_dimensions)
    cloud_labels_arr = decode_jon_cnn_cloud_mask(file_path=labels_path, spatial_dimensions=spatial_dimensions)
    water_labels_arr = decode_jon_cnn_water_mask(file_path=labels_path, spatial_dimensions=spatial_dimensions)
    land_labels_arr = decode_jon_cnn_land_mask(file_path=labels_path, spatial_dimensions=spatial_dimensions)
    

    labels_filename = satobj.capture_name + '-slc'
    labels_nc_filename = labels_filename + '.nc'

    plt.imshow(labels_arr)
    plt.savefig(os.path.join(Path(l1d_nc_path).parent, labels_filename + '_labels.png'))
    plt.close()

    plt.imshow(cloud_labels_arr)
    plt.savefig(os.path.join(Path(l1d_nc_path).parent, labels_filename + '_labels_cloud.png'))
    plt.close()

    plt.imshow(water_labels_arr)
    plt.savefig(os.path.join(Path(l1d_nc_path).parent, labels_filename + '_labels_water.png'))
    plt.close()

    plt.imshow(land_labels_arr)
    plt.savefig(os.path.join(Path(l1d_nc_path).parent, labels_filename + '_labels_land.png'))
    plt.close()


    
    with Dataset(Path(l1d_nc_path).parent / labels_nc_filename, "w", format="NETCDF4") as ncfile:

        # Define dimensions
        ncfile.createDimension("y", lats.shape[0])
        ncfile.createDimension("x", lats.shape[1])

        # Create variables
        latitudes = ncfile.createVariable("lat", "f4", ("y", "x"))
        longitudes = ncfile.createVariable("lon", "f4", ("y", "x"))
        labels = ncfile.createVariable("labels", "f4", ("y", "x"))

        cloud = ncfile.createVariable("cloud", "f4", ("y", "x"))
        water = ncfile.createVariable("water", "f4", ("y", "x"))
        land = ncfile.createVariable("land", "f4", ("y", "x"))

        # Assign data
        latitudes[:, :] = lats
        longitudes[:, :] = lons
        labels[:, :] = labels_arr
        cloud[:, :] = cloud_labels_arr
        water[:, :] = water_labels_arr
        land[:, :] = land_labels_arr


        # Optional: add metadata
        latitudes.units = "degrees_north"
        longitudes.units = "degrees_east"
        labels.units = "unknown"  # Replace with actual units if known
        cloud.units = "unknown"  # Replace with actual units if known
        water.units = "unknown"  # Replace with actual units if known
        land.units = "unknown"  # Replace with actual units if known
        ncfile.description = "NetCDF file containing SLC labels"
    
        



   
        




if __name__ == "__main__":

    if True:
        if len(sys.argv) < 2 or len(sys.argv) > 2:
            print("Usage: python process_l1d_dir.py <nc_dir_path>")
            sys.exit(1)

        dir_path = sys.argv[1]
    else:
        dir_path = '/home/cameron/captures_test/image61N5E_2025-04-02T10-44-29Z'

    base_path = dir_path.rstrip('/')

    folder_name = os.path.basename(base_path)
    #l1a_nc_path = os.path.join(base_path, f"{folder_name}-l1a.nc")
    l1d_nc_path = os.path.join(base_path, f"{folder_name}-l1d.nc")
    lats_path = os.path.join(base_path, "processing-temp", "latitudes_indirectgeoref.dat")
    lons_path = os.path.join(base_path, "processing-temp", "longitudes_indirectgeoref.dat") 


    #print(base_path)
    #print(folder_name)
    #print(lat_file)
    #print(lon_file)
    #lats_path = sys.argv[2] if len(sys.argv) == 4 else None
    #lons_path = sys.argv[3] if len(sys.argv) == 4 else None

    main(l1d_nc_path, lats_path, lons_path)

    #dst_dir = "/home/_shared/ARIEL/atmospheric_correction/OC-SMART/OC-SMART_with_HYPSO/L1B/"
    #dst_file = os.path.join(dst_dir, "HYPSO2_HSI_" + str(folder_name) + "-l1d.nc")

    #import shutil
    #shutil.copy2(l1d_nc_path, dst_file)



