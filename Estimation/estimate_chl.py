#!/usr/bin/env python3

#import sys
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso/')

import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path

from hypso import Hypso1
from hypso.write import write_l1c_nc_file
from hypso.classification import decode_jon_cnn_water_mask, decode_jon_cnn_land_mask, decode_jon_cnn_cloud_mask
from hypso.resample import resample_dataarray_kd_tree_nearest

import netCDF4 as nc
from pyresample.geometry import SwathDefinition

from global_land_mask import globe


H1_PLS_MODEL_PATH = '/home/cameron/Projects/plsr-chl-estimation/Estimation/pls_model_c10_h1.pkl'
H2_PLS_MODEL_PATH = '/home/cameron/Projects/plsr-chl-estimation/Estimation/pls_model_c10_h2.pkl'

PLS_MODEL = H1_PLS_MODEL_PATH

MIDNOR_GRID_PATH = "/home/cameron/Projects/plsr-chl-estimation/Estimation/midnor_grid.nc"

PRODUCE_FIGURES = False

def main(l1a_nc_path, labels_path, dst_path, lats_path=None, lons_path=None):


    if not os.path.isfile(l1a_nc_path):
        print(f"Error: The file '{l1a_nc_path}' does not exist.")
        return
    
    if not os.path.isfile(labels_path):
        print(f"Error: The file '{labels_path}' does not exist.")
        return
    
    if lats_path is not None and not os.path.isfile(lats_path):
        print(f"Error: The file '{lats_path}' does not exist.")
        return
    
    if lons_path is not None and not os.path.isfile(lons_path):
        print(f"Error: The file '{lons_path}' does not exist.")
        return
    

    # Process the first file
    print(f"Processing file: {l1a_nc_path}")

    nc_file = Path(l1a_nc_path)

    satobj = Hypso1(path=nc_file, verbose=True)

    satobj.generate_l1b_cube()
    satobj.generate_l1c_cube()
    satobj.generate_l1d_cube()


    if satobj.sensor == 'hypso1_hsi':
        pls_model = H1_PLS_MODEL_PATH
    elif satobj.sensor == 'hypso2_hsi':
        pls_model = H2_PLS_MODEL_PATH
    else:
        exit()


    # Generate PLSR estimates
    X = satobj.l1d_cube[:,:,6:-6]
    X_dims = X.shape
    X = X.to_numpy().reshape(-1,108)

    with open(pls_model, 'rb') as file:
        pls = pickle.load(file)

    Y = pls.predict(X)
    Y = Y.reshape(X_dims[0], X_dims[1], -1)
    Y = Y[:,:,0]

    Y = np.clip(Y, 0, 10)

    chl_hypso = Y

    if PRODUCE_FIGURES:
        plt.imshow(chl_hypso)
        plt.savefig('./chl_hypso.png')
        plt.close()

    # TODO: Apply masks
    land_mask = decode_jon_cnn_land_mask(file_path=labels_path, spatial_dimensions=satobj.spatial_dimensions)
    cloud_mask = decode_jon_cnn_cloud_mask(file_path=labels_path, spatial_dimensions=satobj.spatial_dimensions)

    mask = cloud_mask | land_mask

    chl_hypso[mask] = np.nan

    if PRODUCE_FIGURES:
        plt.imshow(mask)
        plt.savefig('./mask.png')
        plt.close()

    if PRODUCE_FIGURES:
        plt.imshow(chl_hypso)
        plt.savefig('./chl_hypso_masked.png')
        plt.close()


    cut_off = 10
    radius = 20
    temp = chl_hypso
    indexes = np.where(temp > cut_off)
    mask = cloud_mask
    for row, col in zip(indexes[0], indexes[1]):    
        # Define search boundaries
        row_start, row_end = max(0, row - radius), min(mask.shape[0], row + radius + 1)
        col_start, col_end = max(0, col - radius), min(mask.shape[1], col + radius + 1)
        
        # Check and modify if there's a 1 in the surrounding area
        nearby_area = mask[row_start:row_end, col_start:col_end]
        if np.any(nearby_area == 1):
            temp[row, col] = np.nan


    chl_hypso = temp

    if PRODUCE_FIGURES:
        plt.imshow(chl_hypso)
        plt.savefig('./chl_hypso_expanded_masked.png')
        plt.close()


    # Run indirect georeferencing
    if lats_path is not None and lons_path is not None:
        try:

            with open(lats_path, mode='rb') as file:
                file_content = file.read()
            
            lats = np.frombuffer(file_content, dtype=np.float32)

            lats = lats.reshape(satobj.spatial_dimensions)

            with open(lons_path, mode='rb') as file:
                file_content = file.read()
            
            lons = np.frombuffer(file_content, dtype=np.float32)
  
            lons = lons.reshape(satobj.spatial_dimensions)

            #satobj.run_indirect_georeferencing(points_file_path=points_path, flip=False)

            #lats = satobj.latitudes_indirect
            #lons = satobj.longitudes_indirect

        except Exception as ex:
            print(ex)
            print('Indirect georeferencing has failed. Defaulting to direct georeferencing.')

            satobj.run_direct_georeferencing()

            lats = satobj.latitudes
            lons = satobj.longitudes

    else:
        satobj.run_direct_georeferencing()

        lats = satobj.latitudes
        lons = satobj.longitudes

    if PRODUCE_FIGURES:
        plt.imshow(lats)
        plt.savefig('./lats.png')
        plt.close()

    if PRODUCE_FIGURES:
        plt.imshow(lons)
        plt.savefig('./lons.png')
        plt.close()


    # Load midnor grid, create swath
    with nc.Dataset(MIDNOR_GRID_PATH, format="NETCDF4") as f:
        grid_longitudes = f.variables['gridLons'][:]
        grid_latitudes = f.variables['gridLats'][:]

    target_swath = SwathDefinition(lons=grid_longitudes, lats=grid_latitudes)



    # Resample to midnor grid (nearest)
    chl_hypso_resampled = resample_dataarray_kd_tree_nearest(area_def=target_swath,
                                                             data=chl_hypso,
                                                             latitudes=lats,
                                                             longitudes=lons
                                                             )

    # Apply grid land mask
    #grid_land_mask = np.empty(grid_longitudes.shape)

    grid_x_dim, grid_y_dim = grid_longitudes.shape

    for x_idx in range(0,grid_x_dim):
        for y_idx in range(0,grid_y_dim):
    
            grid_lat = grid_latitudes[x_idx, y_idx]
            grid_lon = grid_longitudes[x_idx, y_idx]

            if globe.is_land(grid_lat, grid_lon):
                chl_hypso_resampled[x_idx, y_idx] = np.nan




    # Get ADCS timestamps 
    #adcssamples = getattr(satobj, 'nc_dimensions')["adcssamples"] #.size

    timestamps = getattr(satobj, 'nc_adcs_vars')["timestamps"]


    # Write to NetCDF 
    write_nc(dst_path=dst_path, chl=chl_hypso_resampled, lats=grid_latitudes, lons=grid_longitudes, timestamps=timestamps)

    if PRODUCE_FIGURES:
        plt.imshow(chl_hypso_resampled)
        plt.savefig('./out.png')
        plt.close()

    if PRODUCE_FIGURES:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        plt.figure(figsize=(16, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([np.min(grid_longitudes), np.max(grid_longitudes), np.min(grid_latitudes), np.max(grid_latitudes)], crs=ccrs.PlateCarree())
        # Plot the resampled data
        mesh = ax.pcolormesh(grid_longitudes, grid_latitudes, chl_hypso_resampled, shading='auto', cmap='viridis', transform=ccrs.PlateCarree())

        # Add basemap 
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

        # Add colorbar and labels
        plt.colorbar(mesh, ax=ax, orientation='vertical', label='Chlorophyll-a (mg/m^3)')
        plt.title('Resampled HYPSO-1 Chlorophyll-a Concentration')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.savefig('./out_decorated.png')

    return chl_hypso, chl_hypso_resampled


def write_nc(dst_path, chl, lats, lons, timestamps):

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

        latitude = netfile.createVariable(
            'latitude', 'f8',
            ('y','x'),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        #latitude.name = "latitude"
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
        #longitude.name = "longitude"
        longitude.standard_name = "longitude"
        longitude.units = "degrees_north"
        longitude[:] = lons


        chlor_a = netfile.createVariable(
            'chl_a', 'f8',
            ('y','x'),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        chlor_a.standard_name = "chl_a"
        chlor_a.units = "mg/m^3" # TODO: check units
        chlor_a[:] = chl


        netfile.createDimension('adcssamples', len(timestamps))

        ts = netfile.createVariable(
            'timestamps', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )

        ts[:] = timestamps

        '''
        # ADCS Timestamps ----------------------------------------------------
        len_timestamps = getattr(satobj, 'nc_dimensions')["adcssamples"] #.size
        netfile.createDimension('adcssamples', len_timestamps)

        meta_adcs_timestamps = netfile.createVariable(
            'metadata/adcs/timestamps', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )

        meta_adcs_timestamps[:] = getattr(satobj, 'nc_adcs_vars')["timestamps"][:]
        '''


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        print("Usage: python script.py <path_to_l1a_nc> <path_to_cnn_labels> <chl_nc_dst> [path_to_lats_file] [path_to_lons_file]")
        sys.exit(1)

    l1a_nc_path = sys.argv[1]

    labels_path = sys.argv[2]

    dst_path = sys.argv[3]

    #points_path = sys.argv[4] if len(sys.argv) == 5 else None

    lats_path = sys.argv[4] if len(sys.argv) == 6 else None
    lons_path = sys.argv[5] if len(sys.argv) == 6 else None

    main(l1a_nc_path, labels_path, dst_path, lats_path, lons_path)


