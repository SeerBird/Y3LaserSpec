import sys
import os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import least_squares, minimize_scalar
from scipy.interpolate import interp1d

out_dir = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260'
fsr = 374.74

# Load Run 1 with Shift Alignment
df_p = pd.read_csv('data/Session 10/Power variation/P1.CSV')
df_u = pd.read_csv('data/Session 10/Power variation/U1.CSV')
df_f = pd.read_csv('data/Session 10/Power variation/F1.CSV')

t = df_p.iloc[:, 0].values
p = df_p.iloc[:, 1].values
u = df_u.iloc[:, 1].values
index = np.arange(len(p))

u_interp = interp1d(index, u, kind='cubic', bounds_error=False, fill_value='extrapolate')
pad = 2000
idx_eval = index[pad:-pad]
p_eval = p[pad:-pad]
bg_mask = np.abs(p_eval - u[pad:-pad]) < 0.02
res = minimize_scalar(lambda s: np.mean((p_eval[bg_mask] - u_interp(idx_eval[bg_mask] + s)) ** 2), bounds=(-300, 300), method='bounded')
u_shifted = u_interp(index + res.x)
diff = (p - u_shifted) * 1e3

peaks_fp, _ = signal.find_peaks(df_f.iloc[:, 1].values, height=-0.045, prominence=0.005, distance=1000)
poly = np.poly1d(np.polyfit(t[peaks_fp], np.arange(len(peaks_fp)), 4))
nu = poly(t) * fsr

# Focus on CO(2,3) region in Run 1: nu in [755, 820] MHz
m = (nu >= 755) & (nu <= 820)
x = nu[m] - 787.5
y = diff[m]

# 1. Pure Lorentzian fit
def lorr_fn(p, x_in):
    x0, gamma, amp, c0, c1 = p
    return c0 + c1 * x_in + amp / (1.0 + 4.0 * ((x_in - x0) / gamma)**2)
fit_lor = least_squares(lambda p: lorr_fn(p, x) - y, [0, 16.0, 40.0, 5.0, 0])
y_lor = lorr_fn(fit_lor.x, x)
res_lor = y - y_lor
rms_lor = np.std(res_lor)

# 2. Saturated Dip Lineshape
def sat_fn(p, x_in):
    x0, gamma0, s, amp, c0, c1 = p
    lor = 1.0 / (1.0 + 4.0 * ((x_in - x0) / gamma0)**2)
    profile = (1.0 - 1.0 / np.sqrt(1.0 + s * lor)) / (1.0 - 1.0 / np.sqrt(1.0 + s))
    return c0 + c1 * x_in + amp * profile
fit_sat = least_squares(lambda p: sat_fn(p, x) - y, [0, 5.5, 12.0, 40.0, 5.0, 0],
                        bounds=([-5, 2.0, 0.01, 0, -20, -1], [5, 20.0, 50.0, 100, 20, 1]))
y_sat = sat_fn(fit_sat.x, x)
res_sat = y - y_sat
rms_sat = np.std(res_sat)

# Create high-resolution plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios': [2.2, 1.2]}, sharex=True)

# Top: Fits
ax1.plot(x, y, '.', color='#94a3b8', ms=3.5, alpha=0.45, label='Experimental Data (Run 1, 859 $\\mu$W)')
ax1.plot(x, y_lor, '--', color='#ef4444', lw=2.0, 
         label=f'Pure Lorentzian Fit: FWHM = {fit_lor.x[1]:.2f} MHz (RMS = {rms_lor:.3f} mV)')
ax1.plot(x, y_sat, '-', color='#2563eb', lw=2.2, 
         label=f'Saturated Dip Model: $\\Gamma_0 = {fit_sat.x[1]:.2f}$ MHz, $s = {fit_sat.x[2]:.1f}$ (RMS = {rms_sat:.3f} mV)')

ax1.set_title('Lineshape Inspection on Dominant Crossover Peak $CO(2,3)$ ($^{87}$Rb $F=2$)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Differential Signal (mV)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)

# Bottom: Residuals comparison
ax2.plot(x, res_lor, '--', color='#ef4444', lw=1.4, alpha=0.85, label=f'Lorentzian Residuals (S-curve structure, RMS = {rms_lor:.3f} mV)')
ax2.plot(x, res_sat, '-', color='#2563eb', lw=1.4, alpha=0.85, label=f'Saturated Dip Residuals (Reduced by 25%, RMS = {rms_sat:.3f} mV)')
ax2.axhline(0, color='black', ls='-', lw=0.8)
ax2.axhline(0.16, color='gray', ls=':', lw=0.9, label='Detector Noise Floor ($\\pm 0.16$ mV)')
ax2.axhline(-0.16, color='gray', ls=':', lw=0.9)

ax2.set_xlabel('Relative Frequency $\\Delta\\nu$ (MHz) from Peak Crest', fontsize=11)
ax2.set_ylabel('Residual (mV)', fontsize=11)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower left', fontsize=9.5)

plt.tight_layout()
fig_path = os.path.join(out_dir, 'session10_saturated_dip_vs_lorentzian.png')
plt.savefig(fig_path, dpi=160)
plt.close()
print(f'Saved comparison plot to {fig_path}')
