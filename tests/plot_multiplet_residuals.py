import sys
import os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import least_squares
import main

fsr = 374.74

# Casimir literature values:
# 87Rb 5P_3/2: A = 84.7185 MHz, B = 12.4965 MHz
E87_all = main.calc_peak_offsets(84.7185, 12.4965, 1.5, 1.5, [1, 2, 3])
offsets_87 = E87_all - E87_all[2]
names_87 = ["F'=1", "F'=2", "F'=3 (Cycling)", "CO(1,2)", "CO(1,3)", "CO(2,3)"]

# 85Rb 5P_3/2: A = 25.0020 MHz, B = 25.7900 MHz
E85_all = main.calc_peak_offsets(25.0020, 25.7900, 2.5, 1.5, [2, 3, 4])
offsets_85 = E85_all - E85_all[2]
names_85 = ["F'=2", "F'=3", "F'=4 (Cycling)", "CO(2,3)", "CO(2,4)", "CO(3,4)"]

def load_and_calibrate(r_id):
    df_p = pd.read_csv(f'data/Session 10/Power variation/P{r_id}.CSV')
    df_u = pd.read_csv(f'data/Session 10/Power variation/U{r_id}.CSV')
    f_path = f'data/Session 10/Power variation/F{r_id}.CSV'
    if not os.path.exists(f_path):
        f_path = 'data/Session 10/Power variation/F14.CSV'
    df_f = pd.read_csv(f_path)
    t = df_p.iloc[:, 0].values
    diff = df_p.iloc[:, 1].values - df_u.iloc[:, 1].values
    fp = df_f.iloc[:, 1].values
    peaks_fp, _ = signal.find_peaks(fp, height=-0.045, prominence=0.005, distance=1000)
    poly = np.poly1d(np.polyfit(t[peaks_fp], np.arange(len(peaks_fp)), 4))
    nu = poly(t) * fsr
    return t, nu, diff

def fit_manifold(nu_man, diff_man, offsets, ref_guess):
    def model_fn(p, x):
        nu_ref, c0, c1 = p[:3]
        gammas = p[3:9]
        amps = p[9:15]
        y = c0 + c1 * (x - nu_ref)
        for k in range(6):
            y += amps[k] / (1.0 + 4.0 * ((x - (nu_ref + offsets[k])) / gammas[k])**2)
        return y
    p0 = [ref_guess, float(np.percentile(diff_man, 5)), 0.0] + [12.0]*6 + [float(np.max(diff_man))/3.0]*6
    b_low = [ref_guess - 40, -0.05, -0.001] + [2.0]*6 + [0.0]*6
    b_high = [ref_guess + 40, 0.05, 0.001] + [40.0]*6 + [1.0]*6
    fit = least_squares(lambda p: model_fn(p, nu_man) - diff_man, p0, bounds=(b_low, b_high), ftol=1e-7, xtol=1e-7)
    return fit.x, model_fn

out_dir = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260'
df_pwr = pd.read_excel('data/Session 10/Power variation/Powers.xlsx')

# ==============================================================================
# PLOT 1: 87Rb Multiplet Fits & Residuals across 4 representative powers
# ==============================================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 14), gridspec_kw={'hspace': 0.38, 'wspace': 0.22})
plot_runs = [1, 5, 7, 21]

for idx, r_id in enumerate(plot_runs):
    pwr = df_pwr['Power_uW'].iloc[r_id-1]
    t, nu, diff = load_and_calibrate(r_id)
    m87_loc = (t >= -0.0035) & (t <= -0.0031)
    t87 = t[m87_loc][np.argmax(diff[m87_loc])]
    nu87_ref = nu[np.argmin(np.abs(t - t87))]
    
    mask = (nu >= nu87_ref - 470) & (nu <= nu87_ref + 30)
    x = nu[mask] - nu87_ref
    y = diff[mask] * 1e3 # in mV
    
    popt, model_fn = fit_manifold(nu[mask], diff[mask], offsets_87, nu87_ref)
    y_fit = model_fn(popt, nu[mask]) * 1e3
    res = y - y_fit
    rms = np.std(res)
    
    # Left: Fit & components
    ax_fit = axes[idx, 0]
    ax_fit.plot(x, y, '.', color='#888888', alpha=0.35, ms=3, label='Data (P - U)')
    ax_fit.plot(x, y_fit, 'r-', lw=1.8, label=f'6-Lorentzian Multiplet Fit (RMS = {rms:.2f} mV)')
    
    # Plot individual decomposed peaks
    colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
    for k in range(6):
        gk = popt[3+k]
        ak = popt[9+k] * 1e3
        x0_k = offsets_87[k] + (popt[0] - nu87_ref)
        comp = ak / (1.0 + 4.0 * ((x - x0_k) / gk)**2)
        ax_fit.plot(x, comp, '--', color=colors[k], lw=1.2, label=f"{names_87[k]}: $\\Gamma={gk:.1f}$ MHz")
        
    ax_fit.set_title(f'Run {r_id} ({pwr:.1f} $\\mu$W): $^{{87}}$Rb $F=2$ Manifold Fit', fontweight='bold', fontsize=11)
    ax_fit.set_ylabel('Signal (mV)', fontsize=10)
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend(loc='upper left', fontsize=7.5, ncol=2)
    
    # Right: Residuals
    ax_res = axes[idx, 1]
    ax_res.plot(x, res, 'k-', lw=0.9, alpha=0.85)
    ax_res.axhline(0, color='red', ls='--', lw=1)
    ax_res.axhline(0.16, color='blue', ls=':', lw=0.8, label='Detector Noise Floor (±0.16 mV)')
    ax_res.axhline(-0.16, color='blue', ls=':', lw=0.8)
    ax_res.set_title(f'Residuals (Data - Fit) [RMS = {rms:.2f} mV]', fontweight='bold', fontsize=11)
    ax_res.set_ylabel('Residual (mV)', fontsize=10)
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(loc='upper right', fontsize=8)

axes[3, 0].set_xlabel('Frequency relative to $F\'=3$ (MHz)', fontsize=11)
axes[3, 1].set_xlabel('Frequency relative to $F\'=3$ (MHz)', fontsize=11)

fig1_path = os.path.join(out_dir, 'session10_multiplet_residuals_87rb.png')
plt.savefig(fig1_path, dpi=160, bbox_inches='tight')
plt.close()
print(f'Saved Figure 1 (87Rb multiplet residuals) to {fig1_path}')

# ==============================================================================
# PLOT 2: 85Rb Multiplet Fits & Residuals across 4 representative powers
# ==============================================================================
fig2, axes2 = plt.subplots(4, 2, figsize=(16, 14), gridspec_kw={'hspace': 0.38, 'wspace': 0.22})

for idx, r_id in enumerate(plot_runs):
    pwr = df_pwr['Power_uW'].iloc[r_id-1]
    t, nu, diff = load_and_calibrate(r_id)
    m85_loc = (t >= 0.00060) & (t <= 0.00075)
    t85 = t[m85_loc][np.argmax(diff[m85_loc])]
    nu85_ref = nu[np.argmin(np.abs(t - t85))]
    
    mask = (nu >= nu85_ref - 220) & (nu <= nu85_ref + 25)
    x = nu[mask] - nu85_ref
    y = diff[mask] * 1e3 # in mV
    
    popt, model_fn = fit_manifold(nu[mask], diff[mask], offsets_85, nu85_ref)
    y_fit = model_fn(popt, nu[mask]) * 1e3
    res = y - y_fit
    rms = np.std(res)
    
    # Left: Fit & components
    ax_fit = axes2[idx, 0]
    ax_fit.plot(x, y, '.', color='#888888', alpha=0.35, ms=3, label='Data (P - U)')
    ax_fit.plot(x, y_fit, 'r-', lw=1.8, label=f'6-Lorentzian Multiplet Fit (RMS = {rms:.2f} mV)')
    
    # Plot individual decomposed peaks
    colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
    for k in range(6):
        gk = popt[3+k]
        ak = popt[9+k] * 1e3
        x0_k = offsets_85[k] + (popt[0] - nu85_ref)
        comp = ak / (1.0 + 4.0 * ((x - x0_k) / gk)**2)
        ax_fit.plot(x, comp, '--', color=colors[k], lw=1.2, label=f"{names_85[k]}: $\\Gamma={gk:.1f}$ MHz")
        
    ax_fit.set_title(f'Run {r_id} ({pwr:.1f} $\\mu$W): $^{{85}}$Rb $F=3$ Manifold Fit', fontweight='bold', fontsize=11)
    ax_fit.set_ylabel('Signal (mV)', fontsize=10)
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend(loc='upper left', fontsize=7.5, ncol=2)
    
    # Right: Residuals
    ax_res = axes2[idx, 1]
    ax_res.plot(x, res, 'k-', lw=0.9, alpha=0.85)
    ax_res.axhline(0, color='red', ls='--', lw=1)
    ax_res.axhline(0.16, color='blue', ls=':', lw=0.8, label='Detector Noise Floor (±0.16 mV)')
    ax_res.axhline(-0.16, color='blue', ls=':', lw=0.8)
    ax_res.set_title(f'Residuals (Data - Fit) [RMS = {rms:.2f} mV]', fontweight='bold', fontsize=11)
    ax_res.set_ylabel('Residual (mV)', fontsize=10)
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(loc='upper right', fontsize=8)

axes2[3, 0].set_xlabel('Frequency relative to $F\'=4$ (MHz)', fontsize=11)
axes2[3, 1].set_xlabel('Frequency relative to $F\'=4$ (MHz)', fontsize=11)

fig2_path = os.path.join(out_dir, 'session10_multiplet_residuals_85rb.png')
plt.savefig(fig2_path, dpi=160, bbox_inches='tight')
plt.close()
print(f'Saved Figure 2 (85Rb multiplet residuals) to {fig2_path}')
