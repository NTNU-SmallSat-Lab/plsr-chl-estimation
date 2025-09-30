from hypso import Hypso
import csv
import subprocess
import os

# Ensure the target directory exists
base_dir = "/home/_shared/ARIEL/PLSR/captures"
os.makedirs(base_dir, exist_ok=True)

# Path to your CSV file
script_dir = os.path.dirname(os.path.realpath(__file__))
csv_file_path = os.path.join(script_dir, "captures.csv")

# Read the CSV and download each file
with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row:  # Skip empty rows
            print(row)
            entry = row[0].strip()
            prefix = entry.split('_')[0]
            url = f"http://129.241.2.147:8008/{prefix}/{entry}/"
            command = [
                "wget", "-r", "-nH", "--cut-dirs=1",
                "-P", base_dir,  # Specify output directory
                url
            ]
            result = subprocess.run(command)
            print(result)

            # If the first attempt fails (non-zero return code), try with port 8009
            if result.returncode != 0:
                print(f"Retrying with port 8009 for {entry}")
                url_8009 = f"http://129.241.2.147:8009/{prefix}/{entry}/"
                command_8009 = [
                    "wget", "-r", "-nH", "--cut-dirs=1",
                    "-P", base_dir,
                    url_8009
                ]
                subprocess.run(command_8009)















'''
# Define the directory path
directory_path = "captures"

# Check if the directory exists
if os.path.isdir(directory_path):
    # List all files ending with '-l1a.nc'
    matching_files = [f for f in os.listdir(directory_path) if f.endswith("-l1a.nc")]
    
    # Print the results
    print("Matching files:")
    for file in matching_files:
        print(file)
else:
    print(f"Directory '{directory_path}' does not exist.")
'''