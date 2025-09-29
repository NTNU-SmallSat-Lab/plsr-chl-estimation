#!/usr/bin/env python3

import os
import sys
import numpy as np

from pathlib import Path
from hypso import Hypso

from hypso.write import write_l1d_nc_file

def main(l1a_nc_path, lats_path=None, lons_path=None):
    # Check if the first file exists
    if not os.path.isfile(l1a_nc_path):
        print(f"Error: The file '{l1a_nc_path}' does not exist.")
        return

    # Process the first file
    print(f"Processing file: {l1a_nc_path}")

    nc_file = Path(l1a_nc_path)

    satobj = Hypso(path=nc_file, verbose=True)

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
        

    write_l1d_nc_file(satobj, overwrite=True, datacube=True)



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


    #print(base_path)
    #print(folder_name)
    #print(lat_file)
    #print(lon_file)
    #lats_path = sys.argv[2] if len(sys.argv) == 4 else None
    #lons_path = sys.argv[3] if len(sys.argv) == 4 else None

    main(l1a_nc_path, lats_path, lons_path)

    #dst_dir = "/home/_shared/ARIEL/atmospheric_correction/OC-SMART/OC-SMART_with_HYPSO/L1B/"
    #dst_file = os.path.join(dst_dir, "HYPSO2_HSI_" + str(folder_name) + "-l1d.nc")

    #import shutil
    #shutil.copy2(l1d_nc_path, dst_file)

