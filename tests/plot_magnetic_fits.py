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
artifact_dir = r'C:\Users\ilyag\.gemini\antigravity\brain\a9066794-3e7c-4948-9556-40d00e7a0260'
os.makedirs(fig_dir, exist_ok=True)

currents = [19, 39, 59, 79, 98]  # mA
fsr = 374.74  # MHz / fringe

def load_file(filepath):
    data = pd.read_csv(filepath).to_numpy().T
    return data[0], data[-1]

def prep_dataset(i):
    fp_path = os.path.join(data_dir, f'GF{i}.CSV')
    p_path = os.path.join(data_dir, f'GP{i}.CSV')
    u_path = os.path.join(data_dir, f'GU{i}.CSV')
    
    t_fp, fabry = load_file(fp_path)
    t_p, pumped = load_file(p_path)
    t_u, unpumped = load_file(u_path)
    
    peaks, _ = find_peaks(fabry, prominence=0.01, distance=100)
    index = np.arange(len(fabry))
    peak_index = np.arange(len(peaks))
    
    pad = 1000
    idx_eval = index[pad:-pad]
    p_eval = pumped[pad:-pad]
    
    res = minimize_scalar(
        lambda s: np.mean((p_eval - np.interp(idx_eval + s, index, unpumped)) ** 2),
        bounds=(-300, 300),
        method='bounded'
    )
    unpumped_aligned = np.interp(index + res.x, index, unpumped)
    diff = pumped - unpumped_aligned
    
    freq_fit = np.poly1d(np.polyfit(peaks, peak_index, 5))
    freq = freq_fit(index)
    
    return freq, diff, fabry, pumped, unpumped_aligned

# Load all 5 datasets
datasets = []
for i in range(1, 6):
    f, d, fabry, p, u = prep_dataset(i)
    datasets.append({
        'index': i,
        'current': currents[i-1],
        'freq': f,
        'diff': d,
        'nu_mhz': f * fsr
    })

def fit_single_peak(nu_axis, y_data, peak_guess_freq, half_span=22.0):
    mask = (nu_axis >= peak_guess_freq - half_span) & (nu_axis <= peak_guess_freq + half_span)
    x = nu_axis[mask]
    y = y_data[mask]
    
    bg_init = float(np.percentile(y, 10))
    amp_init = float(np.max(y) - bg_init)
    x0_init = float(x[np.argmax(y)])
    gamma_init = 15.0
    
    def model(p, x_eval):
        A, x0, gamma, bg, slope = p
        return A / (1.0 + 4.0 * ((x_eval - x0) / gamma) ** 2) + bg + slope * (x_eval - x0)
        
    def res(p):
        return model(p, x) - y
        
    p0 = [amp_init, x0_init, gamma_init, bg_init, 0.0]
    bounds = (
        [0.0, x0_init - 10.0, 3.0, -0.5, -0.05],
        [amp_init * 2.5, x0_init + 10.0, 45.0, 0.5, 0.05]
    )
    fit = least_squares(res, p0, bounds=bounds)
    p_opt = fit.x
    
    J = fit.jac
    dof = len(x) - len(p0)
    mse = np.sum(fit.fun ** 2) / dof
    cov = np.linalg.inv(J.T @ J) * mse
    p_err = np.sqrt(np.diag(cov))
    
    x_dense = np.linspace(x[0], x[-1], 300)
    y_dense = model(p_opt, x_dense)
    residuals = y - model(p_opt, x)
    
    return p_opt, p_err, x, y, residuals, x_dense, y_dense

# Process fits for 87Rb and 85Rb cycling transitions
fits_87 = []
fits_85 = []

for d in datasets:
    # 87Rb F=2 (fringe ~2.8 to 4.6)
    m1_mask = (d['freq'] >= 2.8) & (d['freq'] <= 4.6)
    x_m1 = d['nu_mhz'][m1_mask]
    y_m1 = d['diff'][m1_mask]
    pks, _ = find_peaks(y_m1, prominence=0.03, distance=50)
    peak_nu = x_m1[pks[-1]]
    p_opt, p_err, x, y, resid, x_d, y_d = fit_single_peak(x_m1, y_m1, peak_nu, half_span=22.0)
    fits_87.append({
        'current': d['current'],
        'p_opt': p_opt, 'p_err': p_err,
        'x': x, 'y': y, 'residuals': resid,
        'x_dense': x_d, 'y_dense': y_d
    })
    
    # 85Rb F=3 (fringe ~6.0 to 7.6)
    m2_mask = (d['freq'] >= 6.0) & (d['freq'] <= 7.6)
    x_m2 = d['nu_mhz'][m2_mask]
    y_m2 = d['diff'][m2_mask]
    pks2, _ = find_peaks(y_m2, prominence=0.03, distance=50)
    peak_nu2 = x_m2[pks2[-1]]
    p_opt2, p_err2, x2, y2, resid2, x_d2, y_d2 = fit_single_peak(x_m2, y_m2, peak_nu2, half_span=22.0)
    fits_85.append({
        'current': d['current'],
        'p_opt': p_opt2, 'p_err': p_err2,
        'x': x2, 'y': y2, 'residuals': resid2,
        'x_dense': x_d2, 'y_dense': y_d2
    })

# Plot 1: 5-row x 2-column detailed fit + residual plot for cycling transitions
fig = plt.figure(figsize=(16, 18))
subfigs = fig.subfigures(5, 2, hspace=0.1, wspace=0.15)

for row in range(5):
    cur = currents[row]
    # Col 0: 87Rb F=2 -> F'=3
    sf_87 = subfigs[row, 0]
    axes_87 = sf_87.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    f87 = fits_87[row]
    p = f87['p_opt']
    perr = f87['p_err']
    
    # Main fit
    axes_87[0].plot(f87['x'] - p[1], f87['y'], 'o', ms=2, color='0.4', label='Data', alpha=0.6)
    axes_87[0].plot(f87['x_dense'] - p[1], f87['y_dense'], 'r-', lw=1.5,
                    label=f"Fit: FWHM = {p[2]:.2f} ± {perr[2]:.2f} MHz\nAmp = {p[0]:.3f} V")
    axes_87[0].set_ylabel('Signal (V)', fontsize=10)
    axes_87[0].set_title(f"$^{{87}}$Rb $F=2 \\to F'=3$ at {cur} mA", fontsize=11, fontweight='bold')
    axes_87[0].legend(fontsize=9, loc='upper right')
    axes_87[0].grid(True, alpha=0.3)
    
    # Residuals
    axes_87[1].plot(f87['x'] - p[1], f87['residuals'] * 1e3, 'o', ms=2, color='purple', alpha=0.6)
    axes_87[1].axhline(0, color='k', ls='--', lw=0.8)
    axes_87[1].set_ylabel('Resid (mV)', fontsize=9)
    if row == 4:
        axes_87[1].set_xlabel('Detuning (MHz)', fontsize=10)
    axes_87[1].grid(True, alpha=0.3)
    axes_87[1].set_ylim(-15, 15)
    
    # Col 1: 85Rb F=3 -> F'=4
    sf_85 = subfigs[row, 1]
    axes_85 = sf_85.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    f85 = fits_85[row]
    p2 = f85['p_opt']
    perr2 = f85['p_err']
    
    # Main fit
    axes_85[0].plot(f85['x'] - p2[1], f85['y'], 'o', ms=2, color='0.4', label='Data', alpha=0.6)
    axes_85[0].plot(f85['x_dense'] - p2[1], f85['y_dense'], 'b-', lw=1.5,
                    label=f"Fit: FWHM = {p2[2]:.2f} ± {perr2[2]:.2f} MHz\nAmp = {p2[0]:.3f} V")
    axes_85[0].set_ylabel('Signal (V)', fontsize=10)
    axes_85[0].set_title(f"$^{{85}}$Rb $F=3 \\to F'=4$ at {cur} mA", fontsize=11, fontweight='bold')
    axes_85[0].legend(fontsize=9, loc='upper right')
    axes_85[0].grid(True, alpha=0.3)
    
    # Residuals
    axes_85[1].plot(f85['x'] - p2[1], f85['residuals'] * 1e3, 'o', ms=2, color='teal', alpha=0.6)
    axes_85[1].axhline(0, color='k', ls='--', lw=0.8)
    axes_85[1].set_ylabel('Resid (mV)', fontsize=9)
    if row == 4:
        axes_85[1].set_xlabel('Detuning (MHz)', fontsize=10)
    axes_85[1].grid(True, alpha=0.3)
    axes_85[1].set_ylim(-15, 15)

fig.suptitle('Individual Cycling Peak Fits & Residuals across Coil Currents [19, 39, 59, 79, 98] mA', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(fig_dir, 'cycling_peak_fits_and_residuals.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(artifact_dir, 'cycling_peak_fits_and_residuals.png'), dpi=200, bbox_inches='tight')
plt.close()

# Plot 2: Full Manifold 1 (87Rb F=2) and Manifold 2 (85Rb F=3) side by side for all 5 currents
fig, axes = plt.subplots(5, 2, figsize=(16, 15), sharex='col')

for row in range(5):
    cur = currents[row]
    d = datasets[row]
    
    # Manifold 1 (87Rb F=2)
    m1_mask = (d['freq'] >= 2.8) & (d['freq'] <= 4.6)
    x1 = d['nu_mhz'][m1_mask]
    y1 = d['diff'][m1_mask]
    pks1, _ = find_peaks(y1, prominence=0.03, distance=50)
    x1_ref = x1[pks1[-1]] if len(pks1) > 0 else x1[np.argmax(y1)]
    
    ax1 = axes[row, 0]
    ax1.plot(x1 - x1_ref, y1, color='purple', lw=1.2, label=f'{cur} mA')
    ax1.set_ylabel('Signal (V)', fontsize=10)
    ax1.set_title(f"$^{{87}}$Rb $F=2$ Manifold at {cur} mA", fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-250, 40)
    ax1.legend(loc='upper left', fontsize=9)
    
    # Manifold 2 (85Rb F=3)
    m2_mask = (d['freq'] >= 6.0) & (d['freq'] <= 7.6)
    x2 = d['nu_mhz'][m2_mask]
    y2 = d['diff'][m2_mask]
    pks2, _ = find_peaks(y2, prominence=0.03, distance=50)
    x2_ref = x2[pks2[-1]] if len(pks2) > 0 else x2[np.argmax(y2)]
    
    ax2 = axes[row, 1]
    ax2.plot(x2 - x2_ref, y2, color='teal', lw=1.2, label=f'{cur} mA')
    ax2.set_ylabel('Signal (V)', fontsize=10)
    ax2.set_title(f"$^{{85}}$Rb $F=3$ Manifold at {cur} mA", fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-180, 40)
    ax2.legend(loc='upper left', fontsize=9)

axes[4, 0].set_xlabel('Detuning from Cycling Peak (MHz)', fontsize=11)
axes[4, 1].set_xlabel('Detuning from Cycling Peak (MHz)', fontsize=11)

plt.suptitle('Manifold Multiplets across Solenoid Currents [19, 39, 59, 79, 98] mA', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'manifolds_overview_all_currents.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(artifact_dir, 'manifolds_overview_all_currents.png'), dpi=200, bbox_inches='tight')
plt.close()

print("Saved cycling_peak_fits_and_residuals.png and manifolds_overview_all_currents.png successfully!")
