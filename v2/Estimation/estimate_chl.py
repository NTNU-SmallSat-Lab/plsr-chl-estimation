#!/usr/bin/env python3

#import sys
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso/')

import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path

from hypso import Hypso
from hypso.write import write_l1c_nc_file
from hypso.classification import decode_jon_cnn_water_mask, decode_jon_cnn_land_mask, decode_jon_cnn_cloud_mask
from hypso.resample import resample_dataarray_kd_tree_nearest

import netCDF4 as nc
from pyresample.geometry import SwathDefinition

from global_land_mask import globe



datasets_dir = "/home/_shared/ARIEL/PLSR/datasets"
chl_dir = Path("/home/_shared/ARIEL/PLSR/chlorophyll")
chl_dir.mkdir(parents=True, exist_ok=True)

model_path = os.path.join(datasets_dir, "pls_model_c" + str(16) + ".pkl")

script_dir = os.path.dirname(os.path.realpath(__file__))

MIDNOR_GRID_PATH = os.path.join(script_dir, "midnor_grid.nc")

PRODUCE_FIGURES = True











def write_nc(dst_path, chl, lats, lons, timestamps, grid=True):

    COMP_SCHEME = 'zlib'  # Default: zlib
    COMP_LEVEL = 4  # Default (when scheme != none): 4
    COMP_SHUFFLE = True  # Default (when scheme != none): True

    # Copy dimensions
    if grid:
        with nc.Dataset(MIDNOR_GRID_PATH, format="NETCDF4") as f:
            xc = len(f.dimensions['xc'])
            yc = len(f.dimensions['yc'])
    else:
        xc = lats.shape[1]
        yc = lats.shape[0]


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



















def main(l1a_nc_path, lats_path=None, lons_path=None):
    # Check if the first file exists
    if not os.path.isfile(l1a_nc_path):
        print(f"Error: The file '{l1a_nc_path}' does not exist.")
        return

    # Process the first file
    print(f"Processing file: {l1a_nc_path}")

    nc_file = Path(l1a_nc_path)

    satobj = Hypso(path=nc_file, verbose=True)

    force_reproc = False

    if not os.path.isfile(satobj.l1d_nc_file) or force_reproc:

        #print(satobj.nc_attrs['target_latitude'])
        #print(satobj.nc_attrs['target_longitude'])


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


                # Directly provide the indirect lat/lons loaded from the file. This function will run the track geometry computations.
                satobj.run_indirect_georeferencing(latitudes=lats, longitudes=lons)

                print(satobj.latitudes_indirect)
                print(satobj.longitudes_indirect)

                satobj.generate_l1b_cube()
                satobj.generate_l1c_cube()
                satobj.generate_l1d_cube(use_indirect_georef=True)

            except Exception as ex:
                print(ex)
                print('Indirect georeferencing has failed. Defaulting to direct georeferencing.')

                satobj.run_direct_georeferencing()
                satobj.generate_l1b_cube()
                satobj.generate_l1c_cube()
                satobj.generate_l1d_cube(use_indirect_georef=False)

        else:
            satobj.run_direct_georeferencing()

            satobj.generate_l1b_cube()
            satobj.generate_l1c_cube()
            satobj.generate_l1d_cube(use_indirect_georef=False)

    else:
        satobj = Hypso(path=satobj.l1d_nc_file, verbose=True)


    # Generate PLSR estimates
    X = satobj.l1d_cube[:,:,6:-6]
    X_dims = X.shape
    X = X.to_numpy().reshape(-1,108)

    with open(model_path, 'rb') as file:
        pls = pickle.load(file)

    Y = pls.predict(X)
    Y = Y.reshape(X_dims[0], X_dims[1], -1)
    Y = Y[:,:,0]

    Y = np.clip(Y, 0, 10)

    if PRODUCE_FIGURES:
        plt.imshow(Y)
        plt.savefig('./chl_hypso.png')
        plt.close()


    try:
        lats = satobj.latitudes_indirect
        lons = satobj.longitudes_indirect
    except Exception as ex:
        print(ex)
        lats = satobj.latitudes
        lons = satobj.longitudes

    spatial_dimensions = satobj.spatial_dimensions
    
    full_path = os.path.join(Path(l1a_nc_path).parent, "processing-temp")


    labels_path = os.path.join(full_path, "sea-land-cloud.labels")

    from hypso.classification import decode_jon_cnn_labels, decode_jon_cnn_cloud_mask, decode_jon_cnn_water_mask, decode_jon_cnn_land_mask

    labels_arr = decode_jon_cnn_labels(file_path=labels_path, spatial_dimensions=spatial_dimensions)
    cloud_labels_arr = decode_jon_cnn_cloud_mask(file_path=labels_path, spatial_dimensions=spatial_dimensions)
    water_labels_arr = decode_jon_cnn_water_mask(file_path=labels_path, spatial_dimensions=spatial_dimensions)
    land_labels_arr = decode_jon_cnn_land_mask(file_path=labels_path, spatial_dimensions=spatial_dimensions)
    

    mask = ~water_labels_arr.astype(bool)


    if PRODUCE_FIGURES:
        plt.imshow(mask)
        plt.savefig('./mask.png')
        plt.close()

    Y[mask] = np.nan


    if PRODUCE_FIGURES:
        plt.imshow(Y)
        plt.savefig('./chl_hypso_masked.png')
        plt.close()









    # Load midnor grid, create swath
    with nc.Dataset(MIDNOR_GRID_PATH, format="NETCDF4") as f:
        grid_longitudes = f.variables['gridLons'][:]
        grid_latitudes = f.variables['gridLats'][:]

    target_swath = SwathDefinition(lons=grid_longitudes, lats=grid_latitudes)



    # Resample to midnor grid (nearest)
    Y_resampled = resample_dataarray_kd_tree_nearest(area_def=target_swath,
                                                             data=Y,
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
                Y_resampled[x_idx, y_idx] = np.nan




    # Get ADCS timestamps 
    #adcssamples = getattr(satobj, 'nc_dimensions')["adcssamples"] #.size

    timestamps = getattr(satobj, 'nc_adcs_vars')["timestamps"]



    dst_path = os.path.join(chl_dir, satobj.capture_name + "-plsr-chla.nc")
    
    
    # Write to NetCDF 
    write_nc(dst_path=dst_path, chl=Y_resampled, lats=grid_lat, lons=grid_lon, timestamps=timestamps)




if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Usage: python process_l1d_dir.py <nc_dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    base_path = dir_path.rstrip('/')

    folder_name = os.path.basename(base_path)
    l1a_nc_path = os.path.join(base_path, f"{folder_name}-l1a.nc")
    l1d_nc_path = os.path.join(base_path, f"{folder_name}-l1d.nc")
    lats_path = os.path.join(base_path, "processing-temp", "latitudes_indirectgeoref.dat")
    lons_path = os.path.join(base_path, "processing-temp", "longitudes_indirectgeoref.dat") 


    main(l1a_nc_path, lats_path, lons_path)




















































'''

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

    satobj = Hypso(path=nc_file, verbose=True)

    satobj.generate_l1b_cube()
    satobj.generate_l1c_cube()
    satobj.generate_l1d_cube()

    # Generate PLSR estimates
    X = satobj.l1d_cube[:,:,6:-6]
    X_dims = X.shape
    X = X.to_numpy().reshape(-1,108)

    with open(model_path, 'rb') as file:
        pls = pickle.load(file)

    Y = pls.predict(X)
    Y = Y.reshape(X_dims[0], X_dims[1], -1)
    Y = Y[:,:,0]

    Y = np.clip(Y, 0, 20)

    if PRODUCE_FIGURES:
        plt.imshow(chl_hypso)
        plt.savefig('./chl_hypso.png')
        plt.close()


    exit()

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

'''
    

