"""
Interactive viewer for Protocol 1 unpumped Gaussian absorption dip fits and residuals.
Run with: python show_unpumped_gaussian_fit.py
"""

import sys
sys.path.insert(0, '.')
import main
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Load Protocol 1
fp_file = 'data/Session 9/F1.CSV'
p_file = 'data/Session 9/P1.CSV'
u_file = 'data/Session 9/U1.CSV'

freq, fabry, pumped, unpumped = main.prep_data(fp_file, p_file, u_file, align_shift=False)
fsr = 374.74 # MHz/fringe
nu = freq * fsr

valid = (freq >= 0.5) & (freq <= 26.5)
nu_v = nu[valid]
freq_v = freq[valid]
u_v = unpumped[valid]

# Fit global 4-Gaussian model on cubic baseline
step = 5
nu_fit = nu_v[::step]
u_fit = u_v[::step]

m_bg = ((freq_v[::step] >= 0.5) & (freq_v[::step] <= 2.5)) | \
       ((freq_v[::step] >= 10.0) & (freq_v[::step] <= 12.0)) | \
       ((freq_v[::step] >= 18.0) & (freq_v[::step] <= 19.5)) | \
       ((freq_v[::step] >= 24.5) & (freq_v[::step] <= 26.5))

p_bg_init = np.polyfit(nu_fit[m_bg] * 1e-3, u_fit[m_bg], 3)
dip_centers_init = [1840.0, 2970.0, 5850.0, 8310.0]
amps_init = np.polyval(p_bg_init, np.array(dip_centers_init)*1e-3) - np.array([-1.17, -2.84, -1.42, -0.31])
sigma_init = [320.0, 300.0, 350.0, 320.0]

p0_glob = list(p_bg_init)
for k in range(4):
    p0_glob.extend([amps_init[k], dip_centers_init[k], sigma_init[k]])

def model_4gauss(p, x_mhz):
    c3, c2, c1, c0 = p[:4]
    x_ghz = x_mhz * 1e-3
    y_eval = c3 * x_ghz**3 + c2 * x_ghz**2 + c1 * x_ghz + c0
    for k in range(4):
        Ak, nuk, sigk = p[4 + k*3 : 4 + k*3 + 3]
        y_eval -= Ak * np.exp(-0.5 * ((x_mhz - nuk) / sigk)**2)
    return y_eval

b_low = [-1.0, -1.0, -1.0, -1.0]
b_high = [1.0, 1.0, 1.0, 2.0]
for k in range(4):
    b_low.extend([0.05, dip_centers_init[k] - 300.0, 100.0])
    b_high.extend([5.0, dip_centers_init[k] + 300.0, 600.0])

fit_glob = least_squares(lambda p: model_4gauss(p, nu_fit) - u_fit, p0_glob, bounds=(b_low, b_high))
p_fit = fit_glob.x
y_eval = model_4gauss(p_fit, nu_v)
res = u_v - y_eval

# Figure 1: Full Scan Fit & Residuals
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.0], 'hspace': 0.14})
fig1.canvas.manager.set_window_title('Protocol 1: Full-Scan Unpumped Gaussian Fit')

x_ghz = nu_v * 1e-3
y_bg = p_fit[0]*x_ghz**3 + p_fit[1]*x_ghz**2 + p_fit[2]*x_ghz + p_fit[3]

ax1.plot(x_ghz, u_v, 'k.', markersize=1.0, alpha=0.35, label='Experimental Unpumped Transmission')
ax1.plot(x_ghz, y_eval, 'r-', lw=1.8, label='Global 4-Gaussian Fit')
ax1.plot(x_ghz, y_bg, 'b--', lw=1.5, alpha=0.8, label='Fitted Laser Power Baseline (Cubic)')

dip_names = ['87Rb F=2', '85Rb F=3', '85Rb F=2', '87Rb F=1']
for k in range(4):
    nuk = p_fit[4 + k*3]
    sigk = p_fit[4 + k*3 + 2]
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigk
    ax1.axvline(nuk * 1e-3, color='gray', linestyle=':', alpha=0.6)
    ax1.text(nuk * 1e-3, np.min(u_v)*0.95, f"{dip_names[k]}\n$\\nu_0={nuk:.0f}$ MHz\nFWHM$={fwhm:.0f}$ MHz",
             ha='center', va='bottom', fontsize=8.5, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='gray'))

ax1.set_ylabel('Photodiode Signal (V)', fontsize=11)
ax1.set_title('Protocol 1: Full-Scan Unpumped Doppler Spectrum & Global 4-Gaussian Fit', fontsize=13, fontweight='bold')
ax1.legend(loc='lower left', framealpha=0.9, fontsize=9.5)
ax1.grid(True, linestyle=':', alpha=0.55)

ax2.plot(x_ghz, res, color='crimson', lw=0.8, alpha=0.75, label=f'Residuals (RMS: {np.std(res)*1e3:.1f} mV)')
ax2.axhline(0, color='black', linestyle='--', lw=0.8)
ax2.set_xlabel('Frequency (GHz)', fontsize=11)
ax2.set_ylabel('Residual (V)', fontsize=11)
ax2.set_title('Global Fit Residuals ($y_{data} - y_{fit}$)', fontsize=10.5, fontweight='bold')
ax2.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
ax2.grid(True, linestyle=':', alpha=0.55)

# Figure 2: Detailed Zoomed-in 4 Dips
fig2, axes = plt.subplots(4, 2, figsize=(14, 11), gridspec_kw={'width_ratios': [2.2, 1.0], 'hspace': 0.38, 'wspace': 0.22})
fig2.canvas.manager.set_window_title('Protocol 1: Local Gaussian Fits per Dip')
fig2.suptitle("Protocol 1: Individual Doppler Absorption Dips with Local Gaussian Fits & Residuals", fontsize=12, fontweight='bold')

local_windows = [(1100.0, 2550.0), (2200.0, 3800.0), (5000.0, 6700.0), (7400.0, 9200.0)]
for k in range(4):
    w0, w1 = local_windows[k]
    m_loc = (nu_v >= w0) & (nu_v <= w1)
    x_l = nu_v[m_loc]
    y_l = u_v[m_loc]
    x_mid = 0.5 * (w0 + w1)
    x_rel = x_l - x_mid
    
    def res_l(p):
        b0, b1, b2, A, x0_rel, sig = p
        y_mod = (b0 + b1*(x_rel*1e-3) + b2*(x_rel*1e-3)**2) - A * np.exp(-0.5 * ((x_rel - x0_rel) / sig)**2)
        return y_mod - y_l
        
    p0_l = [np.max(y_l), 0.0, 0.0, np.max(y_l)-np.min(y_l), 0.0, 300.0]
    fit_l = least_squares(res_l, p0_l, bounds=([-2,-5,-5,0.05,-300,100], [2,5,5,5,300,600]))
    b0, b1, b2, A_l, x0_rel_l, sig_l = fit_l.x
    x0_l = x_mid + x0_rel_l
    fwhm_l = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig_l
    y_mod_l = (b0 + b1*(x_rel*1e-3) + b2*(x_rel*1e-3)**2) - A_l * np.exp(-0.5 * ((x_rel - x0_rel_l) / sig_l)**2)
    bg_l = (b0 + b1*(x_rel*1e-3) + b2*(x_rel*1e-3)**2)
    r_l = y_l - y_mod_l

    ax_f, ax_r = axes[k, 0], axes[k, 1]
    ax_f.plot(x_l, y_l, 'k.', markersize=2.0, alpha=0.4, label='Data')
    ax_f.plot(x_l, y_mod_l, 'r-', lw=1.8, label=f"Gaussian Fit (FWHM={fwhm_l:.0f} MHz)")
    ax_f.plot(x_l, bg_l, 'b--', lw=1.2, alpha=0.7, label='Local Baseline')
    ax_f.axvline(x0_l, color='gray', linestyle=':', alpha=0.6)
    ax_f.set_title(f"Dip {k+1}: {dip_names[k]} | Center = {x0_l:.1f} MHz, $\\sigma={sig_l:.1f}$ MHz", fontweight='bold', fontsize=10, loc='left')
    ax_f.set_ylabel('Signal (V)', fontsize=9.5)
    ax_f.legend(loc='lower left', framealpha=0.9, fontsize=8)
    ax_f.grid(True, linestyle=':', alpha=0.5)

    ax_r.plot(x_l, r_l, color='crimson', lw=0.9, alpha=0.8, label=f"Residual (RMS: {np.std(r_l)*1e3:.1f} mV)")
    ax_r.axhline(0, color='black', linestyle='--', lw=0.8)
    ax_r.set_title(f"Residuals ({dip_names[k]})", fontweight='bold', fontsize=10, loc='left')
    ax_r.set_ylabel('Residual (V)', fontsize=9.5)
    ax_r.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax_r.grid(True, linestyle=':', alpha=0.5)

    if k == 3:
        ax_f.set_xlabel('Frequency (MHz)', fontsize=10)
        ax_r.set_xlabel('Frequency (MHz)', fontsize=10)

plt.show()
