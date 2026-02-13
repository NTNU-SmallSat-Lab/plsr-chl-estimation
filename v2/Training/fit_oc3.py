
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import netCDF4 as nc
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso')
sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso1_calibration')
sys.path.insert(0, '/home/_shared/ARIEL/hypso-package/hypso2_calibration')
from hypso import Hypso
from hypso import get_hypso1_wavelengths, get_hypso2_wavelengths

from hypso.calibration import read_coeffs_from_file, \
                              run_radiometric_calibration, \
                              run_destriping_correction, \
                              run_smile_correction


datasets_dir = "/home/_shared/ARIEL/PLSR/datasets_ocx"
dataset_file = os.path.join(datasets_dir, "combined_dataset.h5")
#model_file = os.path.join(datasets_dir, "pls_model_c" + str(components) + ".h5")

# Open the HDF5 file in read mode
with h5py.File(dataset_file, 'r') as h5f:
    # Access datasets
    X = h5f['X'][:]
    Y = h5f['Y'][:]

    # Print shapes
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

#X = X[:, :,6:-6]
#X = X[~mask][:, :,6:-6]
#Y = Y[~mask]

X = np.clip(X, 0, 1)
#Y = np.clip(Y, a_max=1, a_min=-4) # 0.0001 to 10 mg/m3

print(max(Y))
print(min(Y))

plt.hist(10**Y, bins=100)
plt.savefig('Y_hist.png')

#Y = 10**Y
Y = np.clip(Y, 0, 20)
            


h1_wl = get_hypso1_wavelengths()
h2_wl = get_hypso2_wavelengths()

avg_wl = (h1_wl + h2_wl)/2

# Wavelengths are based on PACE wavelengths fro OCI. 
# Indicies are computed from average of H1 and H2 wavelengths,
# both the original 120 bands and a truncated 
# set of 108 bands that drop the first and last 6 bands at each end
# Blue: 442 nm (440.993 nm): 16 (original), 10 (truncated)
# Blue: 443 nm (444.500 nm): 17 (original), 11 (truncated)
# Blue: 490 nm (489.986 nm): 30 (original), 24 (truncated)
# Blue: 510 nm (510.912 nm): 36 (original), 30 (truncated)
# Green: 555 nm (556.108 nm): 49 (original), 43 (truncated) 
# Red: 670 nm (669.947 nm): 82 (original), 76 (truncated)
# Additionally, the CI portion of the algorithm uses the bands
# closest to 443, 555, and 670 nm for blue, green, and red


ocx_wavelength_map = {
    16: 440.993,  
    17: 444.500,
    30: 489.986,   
    36: 510.912,  
    49: 556.108, 
    82: 669.947
}



ocx_wavelength_map_truncated = {
    10: 440.993,  
    11: 444.500,
    24: 489.986,   
    30: 510.912,  
    43: 556.108, 
    76: 669.947
}










#CI uses nearest 443, 555, and 670 nm for blue, green, and red
CI = np.full(X.shape[0], fill_value=np.nan)

a_0_CI = -0.4287
a_1_CI = 230.47

blue_idx = 16
green_idx = 49
red_idx = 82

Rrs_blue = X[:,blue_idx]
Rrs_green = X[:,green_idx]
Rrs_red = X[:,red_idx]

lambda_blue = ocx_wavelength_map[blue_idx]
lambda_green = ocx_wavelength_map[green_idx]
lambda_red = ocx_wavelength_map[red_idx]

CI[:] = Rrs_green - (Rrs_blue + (lambda_green-lambda_blue)/(lambda_red-lambda_blue)*(Rrs_red-Rrs_blue))

chlor_a_CI = 10**(a_0_CI + a_1_CI*CI)


blue_indices = [16, 30, 36]
green_index = 49
rrs_blue1 = X[:,blue_indices[0]]
rrs_blue2 = X[:,blue_indices[1]]
rrs_blue3 = X[:,blue_indices[2]]
rrs_green = X[:,green_index] 

# Calculate ratios for each blue band
ratio1 = rrs_blue1 / rrs_green
ratio2 = rrs_blue2 / rrs_green
ratio3 = rrs_blue3 / rrs_green

# Create masks for ratios > 1
mask1 = ratio1 > 1.0
mask2 = ratio2 > 1.0
mask3 = ratio3 > 1.0

# Initialize output arrays
selected_blue_idx = np.full(X.shape[0], -1, dtype=int)

# Apply selection hierarchy: 442 > 490 > 510
# Start with the longest wavelength (most conservative)
selected_blue_idx[mask3] = blue_indices[2]  # 510 nm

# Override with 490 nm where its ratio > 1 AND it's better than 510
mask_490_better = mask2 & (ratio2 > ratio3)
selected_blue_idx[mask_490_better] = blue_indices[1]

# Override with 442 nm where its ratio > 1 AND it's better than others
mask_442_better = mask1 & (ratio1 > ratio2) & (ratio1 > ratio3)
selected_blue_idx[mask_442_better] = blue_indices[0]



print(set(selected_blue_idx))


selected_green_idx = np.full(selected_blue_idx.shape, fill_value=43)



samples_count = X.shape[0]
per_sample_band_indices = selected_blue_idx

Rrs_blue = X[np.arange(samples_count), selected_blue_idx]
Rrs_green = X[np.arange(samples_count), selected_green_idx]



ratios = Rrs_blue/Rrs_green


print("Stats Rrs_blue")
print(stats.describe(Rrs_blue))

print("Stats Rrs_green")
print(stats.describe(Rrs_green))

print("Stats Y")
print(stats.describe(Y))

print("Stats 10**Y")
print(stats.describe(10**Y))



valid_mask = np.isfinite(ratios) & np.isfinite(Y) #& (10**Y > 0.25) & (ratios > 0)# & (Y > 0)
ratios_masked = ratios[valid_mask]
Y_masked = Y[valid_mask]


R = np.log10(ratios_masked[::100])
T = Y_masked[::100]

R_test = np.log10(ratios_masked[1::100])
T_test = Y_masked[1::100]


print(len(set(R_test)))

print("==== Train set ====")
print(f"Valid data points: {len(R)}")
print(f"Ratio range: {(10**R).min():.3f} to {(10**R).max():.3f}")
print(f"Chl range: {(10**T).min():.3f} to {(10**T).max():.3f} mg/m³")

print("==== Test set ====")
print(f"Valid data points: {len(R_test)}")
print(f"Ratio range: {(10**R_test).min():.3f} to {(10**R_test).max():.3f}")
print(f"Chl range: {(10**T_test).min():.3f} to {(10**T_test).max():.3f} mg/m³")



X_poly = np.column_stack([
        R,                   # R^1
        R**2,                # R^2  
        R**3,                # R^3
        R**4                 # R^4
    ])
    
model = LinearRegression(fit_intercept=True)
model.fit(X_poly, T)


coefficients = {
    'a0': model.intercept_,
    'a1': model.coef_[0],
    'a2': model.coef_[1],
    'a3': model.coef_[2],
    'a4': model.coef_[3]
}



pace_coefficients = {
    'a0': 0.32814,
    'a1': -3.20725,
    'a2': 3.22969,
    'a3': -1.36769,
    'a4': -0.81739
}

print(model.intercept_)
print(model.coef_)





X_poly_test = np.column_stack([
        R_test,                   # R^1
        R_test**2,                # R^2  
        R_test**3,                # R^3
        R_test**4                 # R^4
    ])

T_pred = model.predict(X_poly_test)


T_pred = coefficients['a0'] + \
    coefficients['a1']*R_test + \
    coefficients['a2']*(R_test**2) + \
    coefficients['a3']*(R_test**3) + \
    coefficients['a4']*(R_test**4) 


#eval_mask = (10**T_pred > 0.25) & (10**T_test > 0.25)


#r2 = r2_score(T, T_pred)
#rmse = np.sqrt(mean_squared_error(T, T_pred))

#r2 = r2_score(T[eval_mask], T_pred[eval_mask])
#rmse = np.sqrt(mean_squared_error(T[eval_mask], T_pred[eval_mask]))

chl_truth = T_test
chl_pred = T_pred

print("Stats chl_truth")
print(stats.describe(chl_truth))

print("Stats chl_pred")
print(stats.describe(chl_pred))




r2 = r2_score(chl_truth, chl_pred)
rmse = np.sqrt(mean_squared_error(chl_truth, chl_pred))



print("\n=== OCx Polynomial Fit Results ===")
#print(f"Coefficients: a0={coefficients['a0']:.4f}, "
#        f"a1={coefficients['a1']:.4f}, a2={coefficients['a2']:.4f}, "
#        f"a3={coefficients['a3']:.4f}, a4={coefficients['a4']:.4f}")
print(f"R² score: {r2:.4f}")
#print(f"RMSE (log space): {rmse:.4f}")
#print(f"Approx. relative error: {(10**rmse - 1)*100:.1f}%")
print(f"RMSE: {rmse:.4f}")
print(f"Approx. relative error: {(rmse - 1)*100:.1f}%")   




true_values = chl_truth
predicted_values = chl_pred

plt.figure(figsize=(6, 6))
plt.scatter(true_values, predicted_values, c='crimson', label='Predicted vs True')

# Plot the ideal diagonal line (perfect prediction line)
p1 = max(max(predicted_values), max(true_values))
p2 = min(min(predicted_values), min(true_values))
plt.plot([p1, p2], [p1, p2], 'b-', label='Perfect Prediction')

plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('True Values vs Predicted Values Plot')
plt.axis('equal') # Ensures the x and y axes have the same scale
plt.legend()
plt.show()
plt.savefig('scatter.png')
