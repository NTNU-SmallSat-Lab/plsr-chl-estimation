
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

    y_mask = Y <= 1
    #x_mask = (X < 0) | (X > 0)

    mask = y_mask
    X = X[mask,:]
    Y = Y[mask]

    # Print shapes
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")



X = np.clip(X, 0, 1)
#Y = np.clip(Y, a_max=1, a_min=-4) # 0.0001 to 10 mg/m3 # log10(10) = 1

plt.hist(10**Y, bins=100)
plt.savefig('Y_hist.png')


wl = get_hypso2_wavelengths()


print("======= Computing OCx =======")


blue_1_idx = np.abs(wl - 442).argmin()
blue_2_idx = np.abs(wl - 490).argmin()
blue_3_idx = np.abs(wl - 510).argmin()
green_idx = np.abs(wl - 555).argmin()

lambda_blue_1 = wl[blue_1_idx]
lambda_blue_2 = wl[blue_2_idx]
lambda_blue_3 = wl[blue_3_idx]
lambda_green = wl[green_idx]


Rrs_blue_1 = X[:,blue_1_idx] + 0.0001
Rrs_blue_2 = X[:,blue_2_idx] + 0.0001
Rrs_blue_3 = X[:,blue_3_idx] + 0.0001
Rrs_green = X[:,green_idx] + 0.0001

stacked = np.stack([Rrs_blue_1, Rrs_blue_2, Rrs_blue_3], axis=0)

print("stacked")
print(stacked.shape)

Rrs_blue_max = np.max(stacked, axis=0)


Rrs_blue_1_mask = (Rrs_blue_1 > Rrs_blue_2) & (Rrs_blue_1 > Rrs_blue_3)
Rrs_blue_2_mask = (Rrs_blue_2 > Rrs_blue_1) & (Rrs_blue_2 > Rrs_blue_3)
Rrs_blue_3_mask = (Rrs_blue_3 > Rrs_blue_1) & (Rrs_blue_3 > Rrs_blue_2)


print(Rrs_blue_1_mask)





ratios = Rrs_blue_max / Rrs_green
log_10_ratios = np.log10(ratios)

log_10_chlor_a_truth = Y


print("Ratios min/max")
print(max(ratios))
print(min(ratios))

print("Log10 Ratios min/max")
print(max(log_10_ratios))
print(min(log_10_ratios))


print("Y min/max")
print(max(Y))
print(min(Y))

step = 10000

# Create test set
log_10_ratios_test = log_10_ratios[1::step]
log_10_chlor_a_truth_test = log_10_chlor_a_truth[1::step]


# Create training set
log_10_ratios = log_10_ratios[::step]
log_10_chlor_a_truth = log_10_chlor_a_truth[::step]


coefficients = np.polyfit(log_10_ratios, log_10_chlor_a_truth, 3)

print(coefficients)




# ===== Fit OCx ======

'''
poly_eqn = np.column_stack([
        log_10_ratios,                   # R^1
        log_10_ratios**2,                # R^2  
        log_10_ratios**3,                # R^3
        log_10_ratios**4                 # R^4
    ])
    

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

#coefficients = pace_coefficients

print(coefficients)

#exit()


poly_eqn = np.column_stack([
        log_10_ratios_test,                   # R^1
        log_10_ratios_test**2,                # R^2  
        log_10_ratios_test**3,                # R^3
        log_10_ratios_test**4                 # R^4
    ])


log_10_chlor_a_OCx = model.predict(poly_eqn)

#log_10_chlor_a_OCx = coefficients['a0'] + \
#    coefficients['a1']*log_10_ratios_test + \
#    coefficients['a2']*(log_10_ratios_test**2) + \
#    coefficients['a3']*(log_10_ratios_test**3) + \
#    coefficients['a4']*(log_10_ratios_test**4) 

'''


#log_10_chlor_a_OCx = coefficients[4] + \
#    coefficients[3]*log_10_ratios_test + \
#    coefficients[2]*(log_10_ratios_test**2) + \
#    coefficients[1]*(log_10_ratios_test**3) + \
#    coefficients[0]*(log_10_ratios_test**4) 




#coefficients = [-0.81739, -1.36769, 3.22969, -3.20725, 0.32814]



log_10_chlor_a_OCx = np.polyval(coefficients, log_10_ratios_test)

chlor_a_pred = 10**log_10_chlor_a_OCx 

chlor_a_truth = 10**log_10_chlor_a_truth_test

plt.close()
plt.hist(chlor_a_pred, bins=100)
plt.savefig('Y_hist_chlor_a_pred.png')






plt.close()
plt.figure(figsize=(6, 6))
plt.scatter(chlor_a_pred[Rrs_blue_1_mask[1::step]], 10**log_10_ratios_test[Rrs_blue_1_mask[1::step]], c='crimson', label='Predicted vs True chlor')
plt.show()
plt.savefig('pred_vs_ratio_1.png')














print(len(chlor_a_truth))
print(len(chlor_a_pred))

print("chlor_a_truth")
print(max(chlor_a_truth))
print(min(chlor_a_truth))

print("Stats chl_truth")
print(stats.describe(chlor_a_truth))


print("chlor_a_pred")
print(max(chlor_a_pred))
print(min(chlor_a_pred))

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