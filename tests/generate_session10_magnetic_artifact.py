import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit

def lorentzian(x, x0, gamma, amp, c0, c1):
    return c0 + c1 * (x - x0) + amp / (1.0 + ((x - x0) / gamma)**2)

currents = [0, 10, 20, 30, 40, 50, 60, 70]
fsr = 374.74

fwhm_87_list, err_87_list, amp_87_list = [], [], []
fwhm_85_list, err_85_list, amp_85_list = [], [], []
res_87_std, res_85_std = [], []

fits_87, fits_85 = {}, {}

for i, curr in enumerate(currents, 1):
    df_p = pd.read_csv(f'data/Session 10/Magnetic variation/P{i}.CSV')
    df_u = pd.read_csv(f'data/Session 10/Magnetic variation/U{i}.CSV')
    df_f = pd.read_csv(f'data/Session 10/Magnetic variation/F{i}.CSV')
    
    t = df_p.iloc[:, 0].values
    diff = df_p.iloc[:, 1].values - df_u.iloc[:, 1].values
    fp = df_f.iloc[:, 1].values
    
    peaks_fp, _ = signal.find_peaks(fp, height=-0.045, prominence=0.005, distance=1000)
    poly = np.poly1d(np.polyfit(t[peaks_fp], np.arange(len(peaks_fp)), 4))
    nu = poly(t) * fsr
    
    # 87Rb cycling
    m87_loc = (t >= -0.00335) & (t <= -0.00320)
    t87 = t[m87_loc][np.argmax(diff[m87_loc])]
    nu87 = poly(t87)*fsr
    mask87 = (nu >= nu87 - 25) & (nu <= nu87 + 25)
    x87 = nu[mask87] - nu87
    y87 = diff[mask87]
    popt87, pcov87 = curve_fit(lorentzian, x87, y87, p0=[0.0, 5.0, np.max(y87)-np.min(y87), np.min(y87), 0.0])
    perr87 = np.sqrt(np.diag(pcov87))
    fwhm_87_list.append(2.0 * popt87[1])
    err_87_list.append(2.0 * perr87[1])
    amp_87_list.append(popt87[2] * 1e3)
    res87 = y87 - lorentzian(x87, *popt87)
    res_87_std.append(np.std(res87) * 1e3)
    fits_87[curr] = (x87, y87, popt87, res87)
    
    # 85Rb cycling
    m85_loc = (t >= 0.00074) & (t <= 0.00084)
    t85 = t[m85_loc][np.argmax(diff[m85_loc])]
    nu85 = poly(t85)*fsr
    mask85 = (nu >= nu85 - 20) & (nu <= nu85 + 20)
    x85 = nu[mask85] - nu85
    y85 = diff[mask85]
    popt85, pcov85 = curve_fit(lorentzian, x85, y85, p0=[0.0, 5.0, np.max(y85)-np.min(y85), np.min(y85), 0.0])
    perr85 = np.sqrt(np.diag(pcov85))
    fwhm_85_list.append(2.0 * popt85[1])
    err_85_list.append(2.0 * perr85[1])
    amp_85_list.append(popt85[2] * 1e3)
    res85 = y85 - lorentzian(x85, *popt85)
    res_85_std.append(np.std(res85) * 1e3)
    fits_85[curr] = (x85, y85, popt85, res85)

# Build comprehensive figure
fig = plt.figure(figsize=(15, 12))

# 1. Linewidth vs Current
ax1 = plt.subplot(3, 2, 1)
ax1.errorbar(currents, fwhm_87_list, yerr=err_87_list, fmt='o-', color='#1f77b4', lw=1.5, ms=6, capsize=4, label='$^{87}$Rb ($F=2 \\to F\'=3$)')
ax1.errorbar(currents, fwhm_85_list, yerr=err_85_list, fmt='s--', color='#ff7f0e', lw=1.5, ms=6, capsize=4, label='$^{85}$Rb ($F=3 \\to F\'=4$)')
ax1.axhline(6.07, color='black', ls=':', label='Natural Linewidth (6.07 MHz)')
ax1.set_xlabel('Coil Current (mA)', fontsize=11)
ax1.set_ylabel('FWHM Linewidth (MHz)', fontsize=11)
ax1.set_title('(a) Cycling Line FWHM vs Coil Current', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# 2. Amplitude vs Current
ax2 = plt.subplot(3, 2, 2)
ax2.plot(currents, amp_87_list, 'o-', color='#1f77b4', lw=1.5, ms=6, label='$^{87}$Rb ($F=2 \\to F\'=3$)')
ax2.plot(currents, amp_85_list, 's--', color='#ff7f0e', lw=1.5, ms=6, label='$^{85}$Rb ($F=3 \\to F\'=4$)')
ax2.set_xlabel('Coil Current (mA)', fontsize=11)
ax2.set_ylabel('Peak Amplitude (mV)', fontsize=11)
ax2.set_title('(b) Peak Amplitude vs Coil Current (Dips at 20 mA)', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')

# 3. Fits & Residuals for 87Rb at 0, 20, 30 mA
ax3 = plt.subplot(3, 2, 3)
x_eval = np.linspace(-25, 25, 200)
for c, col in [(0, '#2ca02c'), (20, '#d62728'), (30, '#9467bd')]:
    x_dat, y_dat, popt, _ = fits_87[c]
    ax3.plot(x_dat, y_dat * 1e3, '.', color=col, alpha=0.3, ms=3)
    ax3.plot(x_eval, lorentzian(x_eval, *popt) * 1e3, color=col, lw=1.5, label=f'{c} mA (FWHM={2*popt[1]:.2f} MHz)')
ax3.set_xlabel('Relative Frequency (MHz)', fontsize=11)
ax3.set_ylabel('Signal (mV)', fontsize=11)
ax3.set_title('(c) $^{87}$Rb Cycling Fits (Lorentzian Model)', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

# 4. Residuals for 87Rb
ax4 = plt.subplot(3, 2, 4)
for c, col in [(0, '#2ca02c'), (20, '#d62728'), (30, '#9467bd')]:
    x_dat, _, _, res = fits_87[c]
    ax4.plot(x_dat, res * 1e3, '.', color=col, alpha=0.5, ms=3, label=f'{c} mA (RMS={np.std(res)*1e3:.2f} mV)')
ax4.axhline(0, color='black', lw=0.8, ls=':')
ax4.set_xlabel('Relative Frequency (MHz)', fontsize=11)
ax4.set_ylabel('Fit Residual (mV)', fontsize=11)
ax4.set_title('(d) $^{87}$Rb Fit Residuals (Flat Noise Floor)', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper right')

# 5. Fits & Residuals for 85Rb at 0, 20, 30 mA
ax5 = plt.subplot(3, 2, 5)
for c, col in [(0, '#2ca02c'), (20, '#d62728'), (30, '#9467bd')]:
    x_dat, y_dat, popt, _ = fits_85[c]
    ax5.plot(x_dat, y_dat * 1e3, '.', color=col, alpha=0.3, ms=3)
    ax5.plot(x_eval, lorentzian(x_eval, *popt) * 1e3, color=col, lw=1.5, label=f'{c} mA (FWHM={2*popt[1]:.2f} MHz)')
ax5.set_xlabel('Relative Frequency (MHz)', fontsize=11)
ax5.set_ylabel('Signal (mV)', fontsize=11)
ax5.set_title('(e) $^{85}$Rb Cycling Fits (Lorentzian Model)', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3)
ax5.legend(loc='upper right')

# 6. Residuals for 85Rb
ax6 = plt.subplot(3, 2, 6)
for c, col in [(0, '#2ca02c'), (20, '#d62728'), (30, '#9467bd')]:
    x_dat, _, _, res = fits_85[c]
    ax6.plot(x_dat, res * 1e3, '.', color=col, alpha=0.5, ms=3, label=f'{c} mA (RMS={np.std(res)*1e3:.2f} mV)')
ax6.axhline(0, color='black', lw=0.8, ls=':')
ax6.set_xlabel('Relative Frequency (MHz)', fontsize=11)
ax6.set_ylabel('Fit Residual (mV)', fontsize=11)
ax6.set_title('(f) $^{85}$Rb Fit Residuals', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3)
ax6.legend(loc='upper right')

plt.tight_layout()
out_fig = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260/session10_magnetic_variation_analysis.png'
plt.savefig(out_fig, dpi=160)
print(f'Figure saved to {out_fig}')
