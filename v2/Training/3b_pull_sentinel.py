#!/usr/bin/env python3

import os
import sys
import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
import geopandas as gpd
from pathlib import Path
import requests
import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso')
sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso2_calibration')

#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso1_calibration')
#sys.path.insert(0, '/home/cameron/Projects/hypso-package/hypso2_calibration')

from hypso import Hypso
from hypso.write import write_l1d_nc_file

def grid_to_polygon(lat_matrix, lon_matrix):
    """Convert the external points of lat/lon matrices into a Shapely polygon."""
    # Extract boundary points
    top = list(zip(lon_matrix[0, :], lat_matrix[0, :]))
    right = list(zip(lon_matrix[:, -1], lat_matrix[:, -1]))
    bottom = list(zip(lon_matrix[-1, ::-1], lat_matrix[-1, ::-1]))
    left = list(zip(lon_matrix[::-1, 0], lat_matrix[::-1, 0]))
    # Combine in order and create polygon
    return Polygon(top + right + bottom + left)


def main(l1d_nc_path, lats_path=None, lons_path=None):
    # Check if the first file exists
    if not os.path.isfile(l1d_nc_path):
        print(f"Error: The file '{l1d_nc_path}' does not exist.")
        return

    # Process the first file
    print(f"Processing file: {l1d_nc_path}")

    nc_file = Path(l1d_nc_path)

    satobj = Hypso(path=nc_file, verbose=True)

    print(satobj.nc_attrs)

    from datetime import datetime
    #dt = datetime.fromisoformat(satobj.nc_attrs['start_timestamp_capture'].replace("Z", ""))

    dt = datetime.fromisoformat(satobj.iso_time)
    print("Downloading time:")
    print(dt)

    #print(satobj.latitudes_indirect)
    #print(satobj.longitudes_indirect)

    # Read keys from environment
    #S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
    #S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
    S3_ACCESS_KEY = "sh-1236c23a-30d2-4081-89ed-b5fc70550fd6"
    S3_SECRET_KEY = "sv3fs9eo5hFvCRszi6CcTYMpLiEK1FQu"

    S3_ACCESS_KEY = "TBR6AIXVAL69J9V82M2R"
    S3_SECRET_KEY = "6hOgaruWHSykeBeySSPpFfYcu0ecBU9iKHT40g2N"

    download_folder = nc_file.parent
    download_folder = Path(os.path.join(download_folder, "sentinel_granules"))
    os.makedirs(download_folder, exist_ok=True)

    print(download_folder)


    #grid_lats = satobj.latitudes_indirect
    #grid_lons = satobj.longitudes_indirect

    grid_lats = satobj.latitudes
    grid_lons = satobj.longitudes


    polygon = grid_to_polygon(grid_lats, grid_lons)
    gdf = gpd.GeoDataFrame({'geometry': [polygon]})

    gdf.set_crs('EPSG:32633', inplace=True)  # Example: UTM projection
    gdf = gdf.to_crs('EPSG:4326')
    geometry = [list(coord) for coord in gdf.geometry[0].exterior.coords]
    # Generate a simplified polygon to filter out data that is not in the SINMOD grid 
    simplified_polygon = polygon.simplify(tolerance=0.01, preserve_topology=True)
    # Extract the simplified geometry coordinates
    simplified_geometry = [list(coord) for coord in simplified_polygon.exterior.coords]

    polygon_wkt = "POLYGON((" + ", ".join([f"{lon} {lat}" for lon, lat in simplified_geometry]) + "))"


    # Time window to filter data
    timewindow = timedelta(hours=1.5)
    date_from = (dt - timewindow).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    date_to = (dt + timewindow).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    #print(date_from)
    #print(date_to)


    #Define the OData base URL for Sentinel-3 WFR product type
    odata_base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    # Construct the OData URL with filtering parameters
    odata_url = (
        f"{odata_base_url}?$filter=("
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'OL_2_WFR___')"
        f") and ContentDate/Start ge {date_from} and "
        f"ContentDate/End le {date_to} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon_wkt}')"
        f"&$orderby=PublicationDate desc"
        f"&$expand=Attributes"
    )

    # Making the request to the OData service
    try:
        response = requests.get(odata_url)
        response.raise_for_status()  # Will raise an exception for HTTP errors
        request_data = response.json()
        #print(request_data)
    except requests.exceptions.RequestException as e:
        print(f"Failed to get data: {e}")
        request_data = {}


    if "value" in request_data:
        for result in request_data["value"]:
            try:
                s3_url = result["S3Path"].strip("/eodata/")
                #print(f"Found download URL: {s3_url}")
            except KeyError:
                print("S3Path not found in result.")

    session = boto3.session.Session()
    s3 = session.resource(
        's3',
        endpoint_url='https://eodata.dataspace.copernicus.eu',
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name='default'
    )
    bucket = s3.Bucket("eodata")

    # Files to keep
    files_to_keep = {'chl_nn.nc', 'geo_coordinates.nc', 'tsm_nn.nc', 'wqsf.nc'}
    extracted_folders = set()


    if "value" in request_data:
        for result in request_data["value"]:
            try:
                s3_url = result["S3Path"].strip("/eodata/")

                # ✅ Filter only NR files
                #if "NR" not in s3_url.split("_")[-2]:
                    #print(f"Skipping NT file: {s3_url}")
                    #continue
                print(f"Found download URL: {s3_url}")
                files = bucket.objects.filter(Prefix=s3_url)
                
                print(type(files))
                
                if not list(files):
                    print(f"No files found for: {s3_url}")
                    continue

                for file in files:
                    if file.key and not file.key.endswith("/"):
                        file_name = Path(file.key).name
                        
                        # ✅ Keep only specific files
                        if file_name in files_to_keep:
                            inner_most_folder = Path(file.key).parent.name
                            folder_path = download_folder / inner_most_folder
                            folder_path.mkdir(parents=True, exist_ok=True)

                            local_file_path = folder_path / file_name
                            if not local_file_path.exists():
                                #print(f"Downloading {file.key}...")
                                bucket.download_file(file.key, str(local_file_path))
                                
                                # ✅ Track extracted folders
                                extracted_folders.add(folder_path)
                            else:
                                print(f"File already exists: {local_file_path}")
                        
                print("Downloaded files successfully.")

            except KeyError:
                print("S3Path not found in result.")
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except ClientError as e:
                if e.response['Error']['Code'] == '403':
                    print("Access Denied: Check your S3 credentials and permissions.")
                else:
                    print(f"ClientError: {e}")
            except Exception as e:
                print(f"Error processing file: {e}")



    '''
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

        except Exception as ex:
            print(ex)
            print('No indirect georeferencing files. Defaulting to direct georeferencing.')
    
    '''
        




if __name__ == "__main__":

    if True:
        if len(sys.argv) < 2 or len(sys.argv) > 2:
            print("Usage: python process_l1d_dir.py <nc_dir_path>")
            sys.exit(1)

        dir_path = sys.argv[1]
    else:
        dir_path = '/home/cameron/Nedlastinger/image61N5E_2025-04-02T10-44-29Z'

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



