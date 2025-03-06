from mne.decoding import CSP
import numpy as np

from scipy.signal import welch

import pywt

from statsmodels.tsa.ar_model import AutoReg

def extract_csp_features(X, y, n_components=4):
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    return csp.fit_transform(X, y)



def extract_band_power(X, fs, bands, window_sec=1.0):
    features = []
    for trial in X:
        _, psd = welch(trial, fs, nperseg=int(window_sec*fs))
        band_power = []
        for low, high in bands:
            band_power.append(np.sum(psd[(low <= fs) & (fs <= high)]))
        features.append(band_power)
    return np.array(features)

# Usage
# bands = [(8, 12), (12, 30)]  # Alpha and Beta bands
# bp_features = extract_band_power(X, fs=250, bands=bands)

def extract_wavelet_features(X, wavelet='db4', level=5):
    features = []
    for trial in X:
        coeffs = pywt.wavedec(trial, wavelet, level=level)
        features.append(np.concatenate([np.std(c) for c in coeffs]))
    return np.array(features)

def extract_ar_features(X, order=4):
    features = []
    for trial in X:
        ar_coeffs = []
        for channel in trial:
            model = AutoReg(channel, lags=order).fit()
            ar_coeffs.extend(model.params[1:])  # Exclude constant term
        features.append(ar_coeffs)
    return np.array(features)

def extract_mav_features(X):
    return np.mean(np.abs(X), axis=2)

def extract_combined_features(X, y, fs, bands, wavelet='db4', level=5, ar_order=4):
    csp_features = extract_csp_features(X, y)
    bp_features = extract_band_power(X, fs, bands)
    wavelet_features = extract_wavelet_features(X, wavelet, level)
    ar_features = extract_ar_features(X, ar_order)
    mav_features = extract_mav_features(X)
    
    return np.hstack([csp_features, bp_features, wavelet_features, ar_features, mav_features])