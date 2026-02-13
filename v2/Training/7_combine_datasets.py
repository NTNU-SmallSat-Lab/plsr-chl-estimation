#!/usr/bin/env python3

import os
import glob
import pickle
import h5py

# Path to the datasets directory
datasets_dirs = ["/home/_shared/ARIEL/PLSR/datasets_h1_ocx",
                "/home/_shared/ARIEL/PLSR/datasets_h2_ocx",
                "/home/_shared/ARIEL/PLSR/datasets_ocx"
            ]


for datasets_dir in datasets_dirs:

    print("Combining datasets in " + str(datasets_dir))

    os.makedirs(datasets_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.realpath(__file__))

    pattern = os.path.join(datasets_dir, "*_dataset_*.pkl")
    dataset_paths = glob.glob(pattern)

    print(dataset_paths)



    # Output HDF5 file
    output_file = os.path.join(datasets_dir, "combined_dataset.h5")

    # Initialize HDF5 file
    with h5py.File(output_file, 'w') as h5f:
        X_dset = None
        Y_dset = None

        for filename in dataset_paths:
            if filename.endswith('.pkl'):
                file_path = os.path.join(datasets_dir, filename)
                with open(file_path, 'rb') as f:
                    dataset = pickle.load(f)
                    X = dataset['X']
                    Y = dataset['Y']

                    if X_dset is None:
                        # Create datasets with unlimited size along the first axis
                        X_dset = h5f.create_dataset('X', data=X, maxshape=(None, X.shape[1]), chunks=True)
                        Y_dset = h5f.create_dataset('Y', data=Y, maxshape=(None,), chunks=True)
                    else:
                        print(X.shape[0])
                        if X.shape[0] > 0:
                            # Resize and append
                            X_dset.resize(X_dset.shape[0] + X.shape[0], axis=0)
                            X_dset[-X.shape[0]:] = X

                            Y_dset.resize(Y_dset.shape[0] + Y.shape[0], axis=0)
                            Y_dset[-Y.shape[0]:] = Y
                        else:
                            print("Size is zero. Skipping.")

    print(f"Saved concatenated dataset to {output_file}")

















