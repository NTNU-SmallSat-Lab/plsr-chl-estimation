
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


datasets_dir = "/home/_shared/ARIEL/PLSR/datasets_h2_ocx"
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
Y = np.clip(Y, a_max=1, a_min=-4) # 0.0001 to 10 mg/m3

#Y = np.clip(Y, 0, 1) # log10(10) = 1

print(max(10**Y))
print(min(10**Y))

plt.hist(10**Y, bins=100)
plt.savefig('Y_hist.png')



            


#h1_wl = get_hypso1_wavelengths()
#h2_wl = get_hypso2_wavelengths()
#avg_wl = (h1_wl + h2_wl)/2

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

#ocx_wavelength_map = {
#    16: 440.993,  
#    17: 444.500,
#    30: 489.986,   
#    36: 510.912,  
#    49: 556.108, 
#    82: 669.947
#}

#ocx_wavelength_map_truncated = {
#    10: 440.993,  
#    11: 444.500,
#    24: 489.986,   
#    30: 510.912,  
#    43: 556.108, 
#    76: 669.947
#}

wl = get_hypso2_wavelengths()

blue_idx = np.abs(wl - 442).argmin()
green_idx = np.abs(wl - 555).argmin()
red_idx = np.abs(wl - 670).argmin()

lambda_blue = wl[blue_idx]
lambda_green = wl[green_idx]
lambda_red = wl[red_idx]

print(blue_idx)
print(green_idx)
print(red_idx)


print("Computing CI")


#CI uses nearest 443, 555, and 670 nm for blue, green, and red
CI = np.full(X.shape[0], fill_value=np.nan)

a_0_CI = -0.4287
a_1_CI = 230.47


Rrs_blue = X[:,blue_idx] + 0.0001
Rrs_green = X[:,green_idx] + 0.0001
Rrs_red = X[:,red_idx] + 0.0001

#Rrs_blue = np.clip(Rrs_blue, 0.0001, 0.3)
#Rrs_green = np.clip(Rrs_green, 0.0001, 0.3)
#Rrs_red = np.clip(Rrs_red, 0.0001, 0.3)

CI[:] = Rrs_green - ((Rrs_blue + ((lambda_green-lambda_blue)/(lambda_red-lambda_blue)) * (Rrs_red-Rrs_blue)))

CI = np.clip(CI, -0.004, 0.004)

chlor_a_CI = 10**(a_0_CI + a_1_CI*CI)

chlor_a_s3 = 10**Y

print("Stats chlor_a_CI")
print(stats.describe(chlor_a_CI))

print("Stats chlor_a_s3")
print(stats.describe(chlor_a_s3))


true_values = chlor_a_s3[::1000]
predicted_values = chlor_a_CI[::1000]

plt.figure(figsize=(6, 6))
plt.scatter(true_values, predicted_values, c='crimson', label='Predicted vs True chlor')
p1 = max(max(predicted_values), max(true_values))
p2 = min(min(predicted_values), min(true_values))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlim((0,10))
plt.ylim((0,10))
plt.xlabel('True chlor_a', fontsize=12)
plt.ylabel('Predicted chlor_a', fontsize=12)
plt.title('True chlor_a vs Predicted chlor_a Plot')
#plt.axis('equal') # Ensures the x and y axes have the same scale
plt.legend()
plt.show()
plt.savefig('scatter_ci.png')


true_values = chlor_a_s3[::1000]
predicted_values = chlor_a_CI[::1000]

ci_valid_indices = np.where(predicted_values <= 0.5)


true_values = true_values[ci_valid_indices]
predicted_values = predicted_values[ci_valid_indices]

print(len(true_values))
print(len(predicted_values))

plt.figure(figsize=(6, 6))
plt.scatter(true_values, predicted_values, c='crimson', label='Predicted vs True chlor')
p1 = max(max(predicted_values), max(true_values))
p2 = min(min(predicted_values), min(true_values))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlim((0,3))
plt.ylim((0,1))
plt.xlabel('True chlor_a', fontsize=12)
plt.ylabel('Predicted chlor_a', fontsize=12)
plt.title('True chlor_a vs Predicted chlor_a Plot')
#plt.axis('equal') # Ensures the x and y axes have the same scale
plt.legend()
plt.show()
plt.savefig('scatter_ci_filtered.png')








#exit()











print("======= Computing OCx =======")


blue_1_idx = np.abs(wl - 442).argmin()
blue_2_idx = np.abs(wl - 490).argmin()
blue_3_idx = np.abs(wl - 510).argmin()
green_idx = np.abs(wl - 555).argmin()

lambda_blue_1 = wl[blue_idx]
lambda_blue_2 = wl[green_idx]
lambda_blue_3 = wl[red_idx]
lambda_green = wl[green_idx]


Rrs_blue_1 = X[:,blue_1_idx] + 0.0001
Rrs_blue_2 = X[:,blue_2_idx] + 0.0001
Rrs_blue_3 = X[:,blue_3_idx] + 0.0001
Rrs_green = X[:,green_idx] + 0.0001

stacked = np.stack([Rrs_blue_1, Rrs_blue_2, Rrs_blue_3], axis=0)
Rrs_blue_max = np.max(stacked, axis=0)


# Calculate ratios for each blue band
#ratio1 = Rrs_blue_1 / Rrs_green
#ratio2 = Rrs_blue_2 / Rrs_green
#ratio3 = Rrs_blue_3 / Rrs_green

# Create masks for ratios > 1
#mask1 = ratio1 > 1.0
#mask2 = ratio2 > 1.0
#mask3 = ratio3 > 1.0

ratios = Rrs_blue_max / Rrs_green


print("Applying masks")

#valid_mask = np.isfinite(ratios) & np.isfinite(Y) #& (10**Y > 0.25) & (ratios > 0)# & (Y > 0)
#ratios = ratios[valid_mask]
#log_10_chlor_a_truth = Y[valid_mask]
#chlor_a_CI = chlor_a_CI[valid_mask]

log_10_chlor_a_truth = Y
log_10_ratios = np.log10(ratios)




# Create test set
log_10_ratios_test = log_10_ratios[1::100]
chlor_a_CI_test = chlor_a_CI[1::100]
log_10_chlor_a_truth_test = log_10_chlor_a_truth[1::100]


# Create training set
log_10_ratios = log_10_ratios[::100]
chlor_a_CI = chlor_a_CI[::100]
log_10_chlor_a_truth = log_10_chlor_a_truth[::100]




# ===== Fit OCx ======

poly_eqn = np.column_stack([
        log_10_ratios,                   # R^1
        log_10_ratios**2,                # R^2  
        log_10_ratios**3,                # R^3
        log_10_ratios**4                 # R^4
    ])
    

10**log_10_chlor_a_truth

model = LinearRegression(fit_intercept=True)
model.fit(poly_eqn, log_10_chlor_a_truth)

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







def hybrid_ocx_algorithm(coefficients, log_10_ratios, chlor_a_CI):

    #poly_eqn = np.column_stack([
    #        log_10_ratios,                   # R^1
    #        log_10_ratios**2,                # R^2  
    #        log_10_ratios**3,                # R^3
    #        log_10_ratios**4                 # R^4
    #    ])

    #log_10_chlor_a_OCx = model.predict(poly_eqn)

    log_10_chlor_a_OCx = coefficients['a0'] + \
        coefficients['a1']*log_10_ratios + \
        coefficients['a2']*(log_10_ratios**2) + \
        coefficients['a3']*(log_10_ratios**3) + \
        coefficients['a4']*(log_10_ratios**4) 

    chlor_a_OCx = 10**log_10_chlor_a_OCx 

    # Blending thresholds
    t_1 = 0.25 
    t_2 = 0.35

    chlor_a = np.zeros_like(chlor_a_CI)


    # Get indices for each region
    ci_only_idx = np.where(chlor_a_OCx <= t_1) # CI algorithm only
    blend_idx = np.where((chlor_a_OCx > t_1) & (chlor_a_OCx < t_2))  # Blending region
    ocx_only_idx = np.where(chlor_a_OCx >= t_2) # OCx algorithm only 

    chlor_a_blended = (chlor_a_CI * (t_2 - chlor_a_CI)) / (t_2 - t_1) + (chlor_a_OCx * (chlor_a_CI - t_1)) / (t_2 - t_1)

    chlor_a[ci_only_idx] = chlor_a_CI[ci_only_idx]
    chlor_a[blend_idx] = chlor_a_blended[blend_idx]
    chlor_a[ocx_only_idx] = chlor_a_OCx[ocx_only_idx]

    return chlor_a




chlor_a_pred = hybrid_ocx_algorithm(coefficients, log_10_ratios_test, chlor_a_CI_test)

chlor_a_truth = 10**log_10_chlor_a_truth_test


print(len(chlor_a_truth))
print(len(chlor_a_pred))


print(max(chlor_a_truth))

print("Stats chl_truth")
print(stats.describe(chlor_a_truth))

print("Stats chl_pred")
print(stats.describe(chlor_a_pred))




r2 = r2_score(chlor_a_truth, chlor_a_pred)
rmse = np.sqrt(mean_squared_error(chlor_a_truth, chlor_a_pred))



print("\n=== OCx Polynomial Fit Results ===")
#print(f"Coefficients: a0={coefficients['a0']:.4f}, "
#        f"a1={coefficients['a1']:.4f}, a2={coefficients['a2']:.4f}, "
#        f"a3={coefficients['a3']:.4f}, a4={coefficients['a4']:.4f}")
print(f"R² score: {r2:.4f}")
#print(f"RMSE (log space): {rmse:.4f}")
#print(f"Approx. relative error: {(10**rmse - 1)*100:.1f}%")
print(f"RMSE: {rmse:.4f}")
print(f"Approx. relative error: {(rmse - 1)*100:.1f}%")   




true_values = chlor_a_truth
predicted_values = chlor_a_pred

plt.figure(figsize=(6, 6))
plt.scatter(true_values, predicted_values, c='crimson', label='Predicted vs True chlor')

# Plot the ideal diagonal line (perfect prediction line)
p1 = max(max(predicted_values), max(true_values))
p2 = min(min(predicted_values), min(true_values))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlim((0,10))
plt.ylim((0,10))
plt.xlabel('True chlor_a', fontsize=12)
plt.ylabel('Predicted chlor_a', fontsize=12)
plt.title('True chlor_a vs Predicted chlor_a Plot')
#plt.axis('equal') # Ensures the x and y axes have the same scale
plt.legend()
plt.show()
plt.savefig('scatter_oc4.png')


print(coefficients)