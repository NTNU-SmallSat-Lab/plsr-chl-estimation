#!/usr/bin/env python3

import os
import sys

from pathlib import Path
from hypso import Hypso1

from hypso.write import write_l1c_nc_file

def main(file_path1):
    # Check if the first file exists
    if not os.path.isfile(file_path1):
        print(f"Error: The file '{file_path1}' does not exist.")
        return

    # Process the first file
    print(f"Processing file: {file_path1}")

    nc_file = Path(file_path1)

    satobj = Hypso1(path=nc_file, verbose=True)

    satobj.generate_l1c_cube()

    write_l1c_nc_file(satobj, overwrite=True, datacube=False)



if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Usage: python script.py <path_to_file1> [path_to_file2]")
        sys.exit(1)

    file_path1 = sys.argv[1]

    main(file_path1)


