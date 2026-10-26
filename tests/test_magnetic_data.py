import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import least_squares, minimize_scalar
from scipy.interpolate import interp1d

from util import MANIFOLDS_CONFIG, casimir_energy

# Directory
data_dir = r'data\Session 9\Magnetic variation'
fig_dir = r'figures\magnetic_variation'
os.makedirs(fig_dir, exist_ok=True)

currents = [19, 39, 59, 79, 98]  # mA

def load_file(filepath):
    data = pd.read_csv(filepath).to_numpy().T
    if len(data) > 2:
        return data[0], data[-1]
    return data[0], data[1]

def prep_magnetic_data(i):
    fp_path = os.path.join(data_dir, f'GF{i}.CSV')
    p_path = os.path.join(data_dir, f'GP{i}.CSV')
    u_path = os.path.join(data_dir, f'GU{i}.CSV')
    
    t_fp, fabry = load_file(fp_path)
    t_p, pumped = load_file(p_path)
    t_u, unpumped = load_file(u_path)
    
    # Fabry-perot peaks
    peaks, _ = find_peaks(fabry, prominence=0.01, distance=100)
    index = np.arange(len(fabry))
    peak_index = np.arange(len(peaks))
    
    # Align pumped and unpumped
    u_interp = interp1d(index, unpumped, kind='cubic', bounds_error=False, fill_value='extrapolate')
    pad = 500
    idx_eval = index[pad:-pad]
    p_eval = pumped[pad:-pad]
    bg_mask = np.abs(p_eval - unpumped[pad:-pad]) < 0.04
    if np.sum(bg_mask) < 1000:
        bg_mask = np.ones_like(p_eval, dtype=bool)
    res = minimize_scalar(
        lambda s: np.mean((p_eval[bg_mask] - u_interp(idx_eval[bg_mask] + s)) ** 2),
        bounds=(-200, 200),
        method='bounded'
    )
    unpumped_aligned = u_interp(index + res.x)
    diff = pumped - unpumped_aligned
    
    # Polynomial calibration
    freq_fit = np.poly1d(np.polyfit(peaks, peak_index, 5))
    freq = freq_fit(index)
    
    return freq, diff, peaks

print("Testing loading and calibration for all 5 magnetic variation files...")
for idx, cur in enumerate(currents, start=1):
    freq, diff, peaks = prep_magnetic_data(idx)
    print(f"Dataset {idx} ({cur} mA): {len(peaks)} FP fringes found, freq range [{freq.min():.1f}, {freq.max():.1f}], diff range [{diff.min():.4f}, {diff.max():.4f}]")
