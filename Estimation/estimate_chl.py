#!/usr/bin/env python3

import os
import sys
import pickle

from pathlib import Path
from hypso import Hypso1

from hypso.write import write_l1c_nc_file
from hypso.write import write_products_nc_file

from hypso.classification import decode_jon_cnn_water_mask, decode_jon_cnn_land_mask, decode_jon_cnn_cloud_mask

import netCDF4 as nc
from pyresample.geometry import SwathDefinition

PLS_MODEL_PATH = '/home/cameron/Projects/plsr-chl-estimation/Training/dataset/pls_model_c10.pkl'
MIDNOR_GRID_PATH = "/home/cameron/Projects/plsr-chl-estimation/Estimation/midnor_grid.nc"


def main(l1a_nc_path, labels_path, dst_path, points_path=None):


    if not os.path.isfile(l1a_nc_path):
        print(f"Error: The file '{l1a_nc_path}' does not exist.")
        return
    
    if not os.path.isfile(labels_path):
        print(f"Error: The file '{labels_path}' does not exist.")
        return
    
    if points_path is not None and not os.path.isfile(points_path):
        print(f"Error: The file '{points_path}' does not exist.")
        return
    

    # Process the first file
    print(f"Processing file: {l1a_nc_path}")

    nc_file = Path(l1a_nc_path)

    satobj = Hypso1(path=nc_file, verbose=True)

    satobj.generate_l1b_cube()
    satobj.generate_l1c_cube()
    satobj.generate_l1d_cube()

    #write_l1c_nc_file(satobj, overwrite=True, datacube=False)

    X = satobj.l1d_cube[:,:,6:]
    X_dims = X.shape
    X = X.to_numpy().reshape(-1,114)

    with open(PLS_MODEL_PATH, 'rb') as file:
        pls = pickle.load(file)

    Y = pls.predict(X)
    Y = Y.reshape(X_dims[0], X_dims[1], -1)
    Y = Y[:,:,0]


    chl_hypso = 10**Y

    # TODO: Apply masks
    #land_mask = decode_jon_cnn_land_mask(file_path=labels_path, spatial_dimensions=satobj.spatial_dimensions)
    #cloud_mask = decode_jon_cnn_cloud_mask(file_path=labels_path, spatial_dimensions=satobj.spatial_dimensions)


    # Run indirect georeferencing
    if points_path is not None:
        try:
            satobj.run_indirect_georeferencing(points_file_path=points_path)
        except Exception as ex:
            print(ex)
            print('Indirect georeferencing has failed. Defaulting to direct georeferencing.')

    # Load midnor grid, create swath
    with nc.Dataset(MIDNOR_GRID_PATH, format="NETCDF4") as f:
        grid_longitudes = f.variables['gridLons'][:]
        grid_latitudes = f.variables['gridLats'][:]

    target_swath = SwathDefinition(lons=grid_longitudes, lats=grid_latitudes)


    # Resample to midnor grid (nearest)
    from hypso.resample import resample_dataarray_kd_tree_nearest


    # TODO: add check for indirect or direct lat/lons
    resampled_chl_hypso = resample_dataarray_kd_tree_nearest(area_def=target_swath, 
                                    data=chl_hypso,
                                    latitudes=satobj.latitudes_indirect,
                                    longitudes=satobj.longitudes_indirect
                            )

    # TODO: Write to NetCDF (use spring 2024 NetCDF writer)
    #write_products_nc_file(satobj=satobj, file_name="./chl.nc", overwrite=True)


    write_nc(dst_path='./chlor_a.nc', 
             )


def write_nc(dst_path, chl, lats, lons, ):

    COMP_SCHEME = 'zlib'  # Default: zlib
    COMP_LEVEL = 4  # Default (when scheme != none): 4
    COMP_SHUFFLE = True  # Default (when scheme != none): True

    # Copy dimensions
    with nc.Dataset(MIDNOR_GRID_PATH, format="NETCDF4") as f:
        xc = len(f.dimensions['xc'])
        yc = len(f.dimensions['yc'])

    # Create new NetCDF file
    with (nc.Dataset(dst_path, 'w', format='NETCDF4') as netfile):

        #set_or_create_attr(netfile, attr_name="processing_level", attr_value="L1B")

        # Create dimensions
        netfile.createDimension('y', yc)
        netfile.createDimension('x', xc)


        chlor_a = netfile.createVariable(
            'chlor_a', 'f8',
            ('y','x'),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        chlor_a.long_name = "chlor_a"
        chlor_a.units = "mg/m^3" # TODO: check units
        chlor_a.coordinates = "latitude longitude"
        chlor_a[:] = chl

        latitude = netfile.createVariable(
            'latitude', 'f8',
            ('y','x'),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        latitude.name = "latitude"
        latitude.standard_name = "latitude"
        latitude.units = "degrees_north"
        latitude[:] = lats


        longitude = netfile.createVariable(
            'longitude', 'f8',
            ('y','x'),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        longitude.name = "longitude"
        longitude.standard_name = "longitude"
        longitude.units = "degrees_north"
        longitude[:] = lons

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <path_to_l1a_nc> <path_to_cnn_labels> <chl_nc_dst> [path_to_points_file]")
        sys.exit(1)

    l1a_nc_path = sys.argv[1]

    labels_path = sys.argv[2]

    dst_path = sys.argv[3]

    points_path = sys.argv[4] if len(sys.argv) == 5 else None

    main(l1a_nc_path, labels_path, dst_path, points_path)


