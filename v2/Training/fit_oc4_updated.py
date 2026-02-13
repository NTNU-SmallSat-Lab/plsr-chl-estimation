import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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



BLUE_BANDS=[442, 490, 510]
GREEN_BANDS=[555]



def ocx_model(x, a0, a1, a2, a3, a4):
    """
    OCx polynomial model (usually 4th or 3rd order)
    x = log10(Rrs_blue / Rrs_green)
    Returns log10(Chl_a)
    """
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4

def find_band_indices(wl, target_bands, tolerance=10):
    """
    Find indices of bands closest to target wavelengths
    """
    indices = []
    for target in target_bands:
        idx = np.argmin(np.abs(wl - target))
        if np.abs(wl[idx] - target) <= tolerance:
            indices.append(idx)
        else:
            raise ValueError(f"No band found near {target} nm (closest: {wl[idx]} nm)")
    return indices

def prepare_ocx_features(X, wl, blue_bands=BLUE_BANDS, green_bands=GREEN_BANDS):
    """
    Prepare OCx input features from reflectance data
    
    Parameters:
    -----------
    X : array (n_samples, n_bands)
        Reflectance data
    wl : array (n_bands,)
        Wavelength labels in nm
    blue_bands : list
        Target blue wavelengths (use average if multiple)
    green_bands : list
        Target green wavelengths (use average if multiple)
    
    Returns:
    --------
    ratios : array (n_samples,)
        log10(Rrs_blue / Rrs_green)
    """


    # Find band indices
    blue_indices = find_band_indices(wl, blue_bands)
    green_indices = find_band_indices(wl, green_bands)
    
    # For OC3/OC4: Use MAXIMUM reflectance among blue bands
    # Take the maximum value across specified blue bands for each sample
    Rrs_blue = np.max(X[:, blue_indices], axis=1)

    # For green band: use single band
    Rrs_green = np.mean(X[:, green_indices], axis=1)
    
    # Calculate ratio (avoid division by zero)
    ratio = Rrs_blue / (Rrs_green + 1e-10)
    
    # Apply log10 (avoid log of zero/negative)
    ratio = np.clip(ratio, 1e-10, None)  # Ensure positive
    log_ratio = np.log10(ratio)
    
    return log_ratio

def fit_ocx(X, wl, Y_log10, order=4, blue_bands=BLUE_BANDS, green_bands=GREEN_BANDS):
    """
    Fit OCx algorithm to data
    
    Parameters:
    -----------
    X : array (n_samples, n_bands)
        Reflectance data
    wl : array (n_bands,)
        Wavelengths
    Y_log10 : array (n_samples,)
        log10(Chl_a) values
    order : int
        Polynomial order (typically 3 or 4)
    blue_bands, green_bands : list
        Wavelengths for blue/green bands
    
    Returns:
    --------
    params : dict with fitted parameters and model
    """
    # Prepare features
    x_data = prepare_ocx_features(X, wl, blue_bands, green_bands)
    
    # Remove NaN/inf values
    valid_mask = np.isfinite(x_data) & np.isfinite(Y_log10)
    x_data_clean = x_data[valid_mask]
    y_data_clean = Y_log10[valid_mask]
    
    if order == 4:
        # Fit 4th order polynomial (standard OC4)
        p0 = [0.32814, -3.20725, 3.22969, -1.36769, -0.81739]  # PACE OC4 initial guess
        
        #p0 = [0.333973333333333, -3.09176166666667, 2.90224083333333, -0.996470833333333, -0.920100833333333]
        #v0 = [0.0081790690969697, 0.334240422578788, 3.98836524355379, 6.30754647591743, 0.985412327190152]
        #std0 = np.sqrt(v0)

        bounds = ([-np.inf]*5, [np.inf]*5)
        #bounds = (p0 - 3*std0, p0 + 3*std0)

        def model_func(x, a0, a1, a2, a3, a4):
            return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4
        
    elif order == 3:
        # Fit 3rd order polynomial (OC3)
        p0 = [0.283, -2.753, 1.457, 0.659] 
        bounds = ([-np.inf]*4, [np.inf]*4)
        #bounds = ([-10]*4, [10]*4)
        
        def model_func(x, a0, a1, a2, a3):
            return a0 + a1*x + a2*x**2 + a3*x**3
    
    else:
        raise ValueError("Order must be 3 (OC3) or 4 (OC4)")

    # Fit the model
    params, params_covariance = curve_fit(
        model_func, 
        x_data_clean, 
        y_data_clean,
        p0=p0[:order+1],
        bounds=bounds,
        maxfev=1000
    )
    
    # Calculate predictions and statistics
    y_pred = model_func(x_data_clean, *params)
    
    if False:
        a = [0.32814, -3.20725, 3.22969, -1.36769, -0.81739]
        a0 = a[0]
        a1 = a[1]
        a2 = a[2]
        a3 = a[3]
        a4 = a[4]

        x = x_data_clean

        y_pred = a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4

    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    r2 = r2_score(y_data_clean, y_pred)
    rmse = np.sqrt(mean_squared_error(y_data_clean, y_pred))

    # Calculate R2
    #residuals = y_data_clean - y_pred
    #ss_res = np.sum(residuals**2)
    #ss_tot = np.sum((y_data_clean - np.mean(y_data_clean))**2)
    #r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate RMSE
    #rmse = np.sqrt(np.mean(residuals**2))
    
    # Results
    results = {
        'params': params,
        'covariance': params_covariance,
        'r_squared': r2,
        'rmse': rmse,
        'model_func': model_func,
        'x_data': x_data_clean,
        'y_data': y_data_clean,
        'y_pred': y_pred,
        'order': order,
        'blue_bands': blue_bands,
        'green_bands': green_bands
    }
    
    return results

def predict_ocx(X, wl, model_params, model_func, blue_bands=BLUE_BANDS, green_bands=GREEN_BANDS):
    """
    Predict chlorophyll-a using fitted OCx model
    """
    #if blue_bands is None:
    #    blue_bands = model_params.get('blue_bands', [443, 490])
    #if green_bands is None:
    #    green_bands = model_params.get('green_bands', [555, 560])
    
    # Prepare features
    x_data = prepare_ocx_features(X, wl, blue_bands, green_bands)
    
    # Predict
    log10_chla_pred = model_func(x_data, *model_params)
    
    # Convert from log10 to actual Chl-a
    chla_pred = 10**log10_chla_pred
    
    return chla_pred, log10_chla_pred






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

    y_mask = Y <= 0.7 #1.3
    #x_mask = (X < 0) | (X > 0)

    mask = y_mask
    X = X[mask,:]
    Y_log10 = Y[mask]


    n_samples = X.shape[0]

    random_indices = np.random.choice(n_samples, size=500, replace=False)
    random_indices = np.sort(random_indices)

    X = X[random_indices, :]
    Y_log10 = Y_log10[random_indices]


    # Print shapes
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y_log10.shape}")







X = np.clip(X, 0, 1)
#Y = np.clip(Y_log10, a_max=1, a_min=-4) # 0.0001 to 10 mg/m3 # log10(10) = 1

plt.hist(10**Y_log10, bins=100)
plt.savefig('Y_hist.png')


wl = get_hypso2_wavelengths()






# Fit OC4 algorithm (4th order polynomial)
results = fit_ocx(X, wl, Y_log10, order=3)

# Print results
print(f"OC{results['order']} Algorithm Results:")
print(f"Parameters: {results['params']}")
print(f"R2: {results['r_squared']:.4f}")
print(f"RMSE: {results['rmse']:.4f}")



# Plot results (linear)
plt.figure(figsize=(10, 6))
plt.scatter(10**results['x_data'], 10**results['y_data'], alpha=0.5, label='Data')
plt.scatter(10**results['x_data'], 10**results['y_pred'], alpha=0.5, label='Predictions')
x_sorted = np.sort(results['x_data'])
y_fit = results['model_func'](x_sorted, *results['params'])
plt.plot(10**x_sorted, 10**y_fit, 'r-', lw=2, label='OCx Fit')
plt.xlabel('Rrs_blue / Rrs_green')
plt.ylabel('Chl_a')
plt.legend()
plt.title(f'OC{results["order"]} Algorithm Fit (R2={results["r_squared"]:.3f})')
plt.savefig('output.png')



# Plot results (log)
plt.figure(figsize=(10, 6))
plt.scatter(results['x_data'], results['y_data'], alpha=0.5, label='Data')
plt.scatter(results['x_data'], results['y_pred'], alpha=0.5, label='Predictions')
x_sorted = np.sort(results['x_data'])
y_fit = results['model_func'](x_sorted, *results['params'])
plt.plot(x_sorted, y_fit, 'r-', lw=2, label='OCx Fit')
plt.xlabel('log10(Rrs_blue / Rrs_green)')
plt.ylabel('log10(Chl_a)')
plt.yscale('log')
plt.legend()
plt.title(f'OC{results["order"]} Algorithm Fit (R2={results["r_squared"]:.3f})')
plt.savefig('output_log.png')












# Make predictions on new data
# X_new: new reflectance data
chla_pred, log10_pred = predict_ocx(X, wl, results['params'], results['model_func'])




print(min(chla_pred))
print(max(chla_pred))


exit()



















































