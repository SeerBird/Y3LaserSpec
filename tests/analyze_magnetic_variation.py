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
    
    # Linear interpolation alignment
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

# Plot 1: Full spectra comparison across the 5 currents
plt.figure(figsize=(14, 7))
colors = plt.cm.plasma(np.linspace(0.1, 0.85, 5))
for d, col in zip(datasets, colors):
    plt.plot(d['freq'], d['diff'], label=f"{d['current']} mA", color=col, alpha=0.85, lw=1.0)
plt.xlabel('Fabry-Pérot Fringe Index', fontsize=12)
plt.ylabel('Differential Transmission (V)', fontsize=12)
plt.title('Rubidium Saturated Absorption Spectra vs. Solenoid Current (Session 9 Magnetic Variation)', fontsize=14, fontweight='bold')
plt.legend(title='Coil Current', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(2.5, 23.0)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'full_spectra_vs_current.png'), dpi=200)
plt.close()

# Adjusted windows in fringe units for Magnetic Variation:
windows = [
    ('87Rb F=2', 2.8, 4.6),
    ('85Rb F=3', 6.0, 7.6),
    ('85Rb F=2', 13.8, 15.6),
    ('87Rb F=1', 20.2, 22.2)
]

# Plot 2: 4-panel comparison of the 4 manifolds across the 5 currents
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.ravel()

for ax_idx, (name, w_min, w_max) in enumerate(windows):
    ax = axes[ax_idx]
    for d, col in zip(datasets, colors):
        mask = (d['freq'] >= w_min) & (d['freq'] <= w_max)
        x = d['freq'][mask] * fsr
        y = d['diff'][mask]
        if len(x) > 0:
            # Shift x so that the primary peak lines up
            pks, _ = find_peaks(y, prominence=0.03, distance=50)
            if len(pks) > 0:
                x_ref = x[pks[-1]]
            else:
                x_ref = x[np.argmax(y)]
            ax.plot(x - x_ref, y, label=f"{d['current']} mA", color=col, lw=1.2)
    ax.set_title(f'Manifold: {name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency relative to dominant peak (MHz)', fontsize=10)
    ax.set_ylabel('Signal (V)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-250, 100)

plt.suptitle('Manifold Profiles vs. Magnetic Field Coil Current (Aligned to Dominant Peak)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'manifolds_vs_current.png'), dpi=200)
plt.close()

# Fit the cycling peaks directly:
# 87Rb F=2 -> F'=3 (rightmost peak of Manifold 1)
# 85Rb F=3 -> F'=4 (rightmost peak of Manifold 2)

def fit_single_peak(nu_axis, y_data, peak_guess_freq, half_span=22.0):
    mask = (nu_axis >= peak_guess_freq - half_span) & (nu_axis <= peak_guess_freq + half_span)
    x = nu_axis[mask]
    y = y_data[mask]
    
    bg_init = float(np.percentile(y, 10))
    amp_init = float(np.max(y) - bg_init)
    x0_init = float(x[np.argmax(y)])
    gamma_init = 15.0
    
    # Model: A / (1 + 4 (x - x0)^2 / gamma^2) + bg + slope * (x - x0)
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
    
    # Calculate uncertainty
    J = fit.jac
    dof = len(x) - len(p0)
    mse = np.sum(fit.fun ** 2) / dof
    cov = np.linalg.inv(J.T @ J) * mse
    p_err = np.sqrt(np.diag(cov))
    
    return p_opt, p_err, x, y, model(p_opt, x)

fit_results_87 = []
fit_results_85 = []

for d in datasets:
    # 87Rb F=2: Find rightmost prominent peak
    m1_mask = (d['freq'] >= 2.8) & (d['freq'] <= 4.6)
    x_m1 = d['nu_mhz'][m1_mask]
    y_m1 = d['diff'][m1_mask]
    pks, _ = find_peaks(y_m1, prominence=0.03, distance=50)
    peak_nu = x_m1[pks[-1]]
    p_opt, p_err, x_fit, y_fit, y_eval = fit_single_peak(x_m1, y_m1, peak_nu, half_span=22.0)
    fit_results_87.append({
        'current': d['current'],
        'amp': p_opt[0], 'amp_err': p_err[0],
        'x0': p_opt[1], 'x0_err': p_err[1],
        'fwhm': p_opt[2], 'fwhm_err': p_err[2],
        'bg': p_opt[3], 'slope': p_opt[4],
        'x_fit': x_fit, 'y_fit': y_fit, 'y_eval': y_eval
    })
    
    # 85Rb F=3: Find rightmost prominent peak
    m2_mask = (d['freq'] >= 6.0) & (d['freq'] <= 7.6)
    x_m2 = d['nu_mhz'][m2_mask]
    y_m2 = d['diff'][m2_mask]
    pks2, _ = find_peaks(y_m2, prominence=0.03, distance=50)
    peak_nu2 = x_m2[pks2[-1]]
    p_opt2, p_err2, x_fit2, y_fit2, y_eval2 = fit_single_peak(x_m2, y_m2, peak_nu2, half_span=22.0)
    fit_results_85.append({
        'current': d['current'],
        'amp': p_opt2[0], 'amp_err': p_err2[0],
        'x0': p_opt2[1], 'x0_err': p_err2[1],
        'fwhm': p_opt2[2], 'fwhm_err': p_err2[2],
        'bg': p_opt2[3], 'slope': p_opt2[4],
        'x_fit': x_fit2, 'y_fit': y_fit2, 'y_eval': y_eval2
    })

df_87 = pd.DataFrame(fit_results_87)
df_85 = pd.DataFrame(fit_results_85)

print("\n--- 87Rb F=2 -> F'=3 (Cycling Transition) ---")
print(df_87[['current', 'amp', 'fwhm', 'fwhm_err']])

print("\n--- 85Rb F=3 -> F'=4 (Cycling Transition) ---")
print(df_85[['current', 'amp', 'fwhm', 'fwhm_err']])

# Plot 3: FWHM and Amplitude vs. Current
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# FWHM
ax1.errorbar(df_87['current'], df_87['fwhm'], yerr=df_87['fwhm_err'], fmt='o-', color='purple', label="87Rb F=2 -> F'=3", capsize=4, lw=1.5)
ax1.errorbar(df_85['current'], df_85['fwhm'], yerr=df_85['fwhm_err'], fmt='s-', color='teal', label="85Rb F=3 -> F'=4", capsize=4, lw=1.5)
ax1.axhline(6.07, color='r', ls='--', label=r'Natural Linewidth $\Gamma_{nat} = 6.07$ MHz')
ax1.set_xlabel('Coil Current (mA)', fontsize=12)
ax1.set_ylabel('Fitted FWHM (MHz)', fontsize=12)
ax1.set_title('Fitted Linewidth (FWHM) vs. Solenoid Current', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Amplitude
ax2.plot(df_87['current'], df_87['amp'], 'o-', color='purple', label="87Rb F=2 -> F'=3", lw=1.5)
ax2.plot(df_85['current'], df_85['amp'], 's-', color='teal', label="85Rb F=3 -> F'=4", lw=1.5)
ax2.set_xlabel('Coil Current (mA)', fontsize=12)
ax2.set_ylabel('Peak Amplitude (V)', fontsize=12)
ax2.set_title('Peak Amplitude vs. Solenoid Current', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fwhm_and_amp_vs_current.png'), dpi=200)
plt.close()

# Plot 4: Peak lineshape overlay for the 5 currents
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for r, col in zip(fit_results_87, colors):
    x_rel = r['x_fit'] - r['x0']
    y_norm = (r['y_fit'] - r['bg']) / r['amp']
    ax1.plot(x_rel, y_norm, label=f"{r['current']} mA", color=col, lw=1.2)
ax1.set_title("87Rb F=2 -> F'=3 Peak Shape vs. Current", fontsize=12, fontweight='bold')
ax1.set_xlabel('Detuning from Peak Center (MHz)', fontsize=11)
ax1.set_ylabel('Normalized Amplitude', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-22, 22)

for r, col in zip(fit_results_85, colors):
    x_rel = r['x_fit'] - r['x0']
    y_norm = (r['y_fit'] - r['bg']) / r['amp']
    ax2.plot(x_rel, y_norm, label=f"{r['current']} mA", color=col, lw=1.2)
ax2.set_title("85Rb F=3 -> F'=4 Peak Shape vs. Current", fontsize=12, fontweight='bold')
ax2.set_xlabel('Detuning from Peak Center (MHz)', fontsize=11)
ax2.set_ylabel('Normalized Amplitude', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-22, 22)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'peak_shapes_vs_current.png'), dpi=200)
plt.close()

print("All plots generated and saved successfully to figures/magnetic_variation/")
