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

    #lats = satobj.latitudes_indirect
    #lons = satobj.longitudes_indirect

    lats = satobj.latitudes
    lons = satobj.longitudes
    
    # Get HYPSO DT
    from datetime import datetime
    #hypso_dt = datetime.fromisoformat(satobj.nc_attrs['timestamp_acquired'].replace("Z", ""))
    hypso_dt = datetime.fromisoformat(satobj.iso_time)
    hypso_dt = hypso_dt.replace(tzinfo=None)

    full_path = os.path.join(Path(l1d_nc_path).parent, "sentinel_granules")


    sentinel_scenes = {}


    for i, entry in enumerate(os.listdir(full_path)):
        
        sentinel_path = os.path.join(full_path, entry)




        # Get Sentinel DT
        filenames = []
        filenames = filenames + glob.glob(sentinel_path + '/geo_coordinates.nc')
        filenames = filenames + glob.glob(sentinel_path + '/chl_nn.nc')

        sentinel_scene = Scene(filenames=filenames, reader='olci_l2')

        sentinel_dt = sentinel_scene.start_time
        sentinel_dt = sentinel_dt.replace(tzinfo=None)

        sentinel_scenes[sentinel_dt] = sentinel_scene




    # Compare Sentinel-3 and HYPSO datetimes. Select closest match

    dates = sentinel_scenes.keys()

    print(hypso_dt)
    print(type(hypso_dt))

    print("Sentinel-3 Matchups under consideration:")
    for date in dates:
        print(date)
        print(type(date))




    closest_dt = min(dates, key=lambda d: abs(d - hypso_dt))

    print("Closest Sentinel-3 matchup:")
    print(closest_dt)

    try:

        sentinel_scene = sentinel_scenes[closest_dt]

        #filenames = []
        #filenames = filenames + glob.glob(sentinel_path + '/geo_coordinates.nc')
        #filenames = filenames + glob.glob(sentinel_path + '/chl_nn.nc')
        #sentinel_scene = Scene(filenames=filenames, reader='olci_l2')
        sentinel_scene.load(['chl_nn'])

        data = sentinel_scene['chl_nn']

        s1 = sentinel_scene.to_xarray(include_lonlats=True)
        src_lats = s1['latitude'].to_numpy()
        src_lons = s1['longitude'].to_numpy()

        src_lats = xr.DataArray(src_lats, dims=['y', 'x'])
        src_lons = xr.DataArray(src_lons, dims=['y', 'x'])

        src_swath_def = SwathDefinition(lons=src_lons, lats=src_lats)

        dst_lons = xr.DataArray(lons, dims=['y', 'x'])
        dst_lats = xr.DataArray(lats, dims=['y', 'x'])

        dst_swath_def = SwathDefinition(lons=dst_lons, lats=dst_lats)

        nnrs = KDTreeNearestXarrayResampler(source_geo_def=src_swath_def, target_geo_def=dst_swath_def)
        sentinel_chl = nnrs.resample(data, fill_value=np.nan, radius_of_influence=500)

        sentinel_chl = sentinel_chl.to_numpy()

        chl_filename = satobj.capture_name + '_sentinel_chl' #'_sentinel_chl_' + str(i)
        chl_nc_filename = chl_filename + '.nc'

        plt.imshow(sentinel_chl)
        #plt.savefig(chl_filename + '.png')
        plt.savefig(os.path.join(Path(l1d_nc_path).parent, chl_filename + '.png'))
        plt.close()


        sentinel_mask = np.isnan(sentinel_chl)

        plt.imshow(sentinel_mask)
        #plt.savefig(os.path.join(full_path, str(folder_name) + "_sentinel_chl_mask_" + str(i) + ".png"))
        plt.savefig(os.path.join(Path(l1d_nc_path).parent, chl_filename + '_mask.png'))
        plt.close()



        with Dataset(Path(l1d_nc_path).parent / chl_nc_filename, "w", format="NETCDF4") as ncfile:


            # Define dimensions
            ncfile.createDimension("y", lats.shape[0])
            ncfile.createDimension("x", lats.shape[1])

            # Create variables
            latitudes = ncfile.createVariable("lat", "f4", ("y", "x"))
            longitudes = ncfile.createVariable("lon", "f4", ("y", "x"))
            chl = ncfile.createVariable("chl_nn", "f4", ("y", "x"))
            mask = ncfile.createVariable("mask", "f4", ("y", "x"))

            # Assign data
            latitudes[:, :] = lats
            longitudes[:, :] = lons
            chl[:, :] = sentinel_chl
            mask[:, :] = sentinel_mask


            # Optional: add metadata
            latitudes.units = "degrees_north"
            longitudes.units = "degrees_east"
            chl.units = "unknown"  # Replace with actual units if known
            mask.units = "unknown"  # Replace with actual units if known
            ncfile.description = "NetCDF file containing resampled Sentinel-3 chl from " + str(entry)


    except Exception as ex:
        print(ex)
        #continue

        



   
        




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



