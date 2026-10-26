import sys
import os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import interp1d

out_dir = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260'
fsr = 374.74

low_runs = [22, 9, 8, 11, 7, 21]
df_pwr = pd.read_excel('data/Session 10/Power variation/Powers.xlsx')

def lorr(x, x0, gamma, amp, c0, c1):
    return c0 + c1 * (x - x0) + amp / (1.0 + 4.0 * ((x - x0) / gamma)**2)

fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
axes = axes.flatten()

for idx, r_id in enumerate(low_runs):
    pwr = df_pwr['Power_uW'].iloc[r_id-1]
    df_p = pd.read_csv(f'data/Session 10/Power variation/P{r_id}.CSV')
    df_u = pd.read_csv(f'data/Session 10/Power variation/U{r_id}.CSV')
    f_path = f'data/Session 10/Power variation/F{r_id}.CSV'
    if not os.path.exists(f_path): f_path = 'data/Session 10/Power variation/F14.CSV'
    df_f = pd.read_csv(f_path)
    
    t = df_p.iloc[:, 0].values
    p = df_p.iloc[:, 1].values
    u = df_u.iloc[:, 1].values
    index = np.arange(len(p))
    
    u_interp = interp1d(index, u, kind='cubic', bounds_error=False, fill_value='extrapolate')
    pad = 2000
    res_s = minimize_scalar(lambda s: np.mean((p[pad:-pad] - u_interp(index[pad:-pad] + s)) ** 2), bounds=(-300, 300), method='bounded')
    diff = (p - u_interp(index + res_s.x)) * 1e3
    
    peaks_fp, _ = signal.find_peaks(df_f.iloc[:, 1].values, height=-0.045, prominence=0.005, distance=1000)
    poly = np.poly1d(np.polyfit(t[peaks_fp], np.arange(len(peaks_fp)), 4))
    nu = poly(t) * fsr
    
    m87 = (t >= -0.0036) & (t <= -0.0031)
    t87 = t[m87][np.argmax(diff[m87])]
    nu87 = poly(t87) * fsr
    
    w = (nu >= nu87 - 14) & (nu <= nu87 + 14)
    x = nu[w] - nu87
    y = diff[w]
    
    popt, pcov = curve_fit(lorr, x, y, p0=[0, 6.0, np.max(y)-np.min(y), np.min(y), 0.0])
    x_dense = np.linspace(-14, 14, 300)
    y_fit = lorr(x_dense, *popt)
    res = y - lorr(x, *popt)
    rms = np.std(res)
    fwhm = popt[1]
    fwhm_err = np.sqrt(pcov[1, 1])
    amp = popt[2]
    
    ax = axes[idx]
    # Plot data points
    ax.plot(x, y, '.', color='#475569', ms=3.5, alpha=0.5, label='Data (P - U_shifted)')
    ax.plot(x_dense, y_fit, 'r-', lw=1.8, label=f'Lorentzian (FWHM = {fwhm:.2f} $\\pm$ {fwhm_err:.2f} MHz)')
    ax.plot(x_dense, popt[3] + popt[4] * (x_dense - popt[0]), 'k:', lw=1.1, label='Baseline')
    
    # Inset for residuals
    ax_ins = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
    ax_ins.plot(x, res, 'k-', lw=0.7, alpha=0.8)
    ax_ins.axhline(0, color='red', ls='--', lw=0.8)
    ax_ins.axhline(0.16, color='blue', ls=':', lw=0.7)
    ax_ins.axhline(-0.16, color='blue', ls=':', lw=0.7)
    ax_ins.set_ylim(-0.6, 0.6)
    ax_ins.set_title(f'Residuals (RMS={rms:.3f} mV)', fontsize=8)
    ax_ins.tick_params(labelsize=7)
    ax_ins.grid(True, alpha=0.2)
    
    ax.set_title(f'Run {r_id} ({pwr:.1f} $\\mu$W): $^{{87}}$Rb Cycling Line ($F=2 \\to F\'=3$)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Relative Frequency $\\Delta\\nu$ (MHz)', fontsize=9.5)
    ax.set_ylabel('Signal (mV)', fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8.5)

fig_path = os.path.join(out_dir, 'session10_low_power_cycling_residuals.png')
plt.savefig(fig_path, dpi=160)
plt.close()
print(f'Saved low power residual inspection to {fig_path}')
