import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit

def lorentzian(x, x0, gamma, amp, c0, c1):
    return c0 + c1 * (x - x0) + amp / (1.0 + ((x - x0) / gamma)**2)

currents_map = {
    1: 0.0, 2: 9.0, 3: 19.0, 4: 29.0, 5: 39.0, 6: 48.5, 7: 59.0, 8: 69.0,
    9: 21.0, 10: 27.0, 11: 32.0, 12: 37.0, 13: 25.0, 14: 16.0
}

fsr = 374.74  # MHz
out_dir = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260'
fits_dir = os.path.join(out_dir, 'session10_fits_14runs')
os.makedirs(fits_dir, exist_ok=True)

results_87 = []
results_85 = []
data_87 = {}
data_85 = {}

for run_id in range(1, 15):
    curr = currents_map[run_id]
    df_p = pd.read_csv(f'data/Session 10/Magnetic variation/P{run_id}.CSV')
    df_u = pd.read_csv(f'data/Session 10/Magnetic variation/U{run_id}.CSV')
    df_f = pd.read_csv(f'data/Session 10/Magnetic variation/F{run_id}.CSV')
    
    t = df_p.iloc[:, 0].values
    diff = df_p.iloc[:, 1].values - df_u.iloc[:, 1].values
    fp = df_f.iloc[:, 1].values
    
    # FP calibration
    peaks_fp, _ = signal.find_peaks(fp, height=-0.045, prominence=0.005, distance=1000)
    poly = np.poly1d(np.polyfit(t[peaks_fp], np.arange(len(peaks_fp)), 4))
    nu = poly(t) * fsr
    
    # --- 87Rb Cycling (F=2 -> F'=3) ---
    m87_loc = (t >= -0.0034) & (t <= -0.00315)
    t87 = t[m87_loc][np.argmax(diff[m87_loc])]
    nu87 = poly(t87) * fsr
    mask87 = (nu >= nu87 - 25) & (nu <= nu87 + 25)
    x87 = nu[mask87] - nu87
    y87 = diff[mask87]
    popt87, pcov87 = curve_fit(lorentzian, x87, y87, p0=[0.0, 4.5, np.max(y87)-np.min(y87), np.min(y87), 0.0])
    perr87 = np.sqrt(np.diag(pcov87))
    fwhm87 = 2.0 * popt87[1]
    err87 = 2.0 * perr87[1]
    res87 = y87 - lorentzian(x87, *popt87)
    
    results_87.append({
        'run': run_id, 'current': curr, 'fwhm': fwhm87, 'err': err87,
        'amp': popt87[2] * 1e3, 'res_rms': np.std(res87) * 1e3
    })
    data_87[curr] = (x87, y87, popt87, res87, fwhm87, err87, run_id)
    
    # --- 85Rb Cycling (F=3 -> F'=4) ---
    m85_loc = (t >= 0.00073) & (t <= 0.00085)
    t85 = t[m85_loc][np.argmax(diff[m85_loc])]
    nu85 = poly(t85) * fsr
    mask85 = (nu >= nu85 - 20) & (nu <= nu85 + 20)
    x85 = nu[mask85] - nu85
    y85 = diff[mask85]
    popt85, pcov85 = curve_fit(lorentzian, x85, y85, p0=[0.0, 4.5, np.max(y85)-np.min(y85), np.min(y85), 0.0])
    perr85 = np.sqrt(np.diag(pcov85))
    fwhm85 = 2.0 * popt85[1]
    err85 = 2.0 * perr85[1]
    res85 = y85 - lorentzian(x85, *popt85)
    
    results_85.append({
        'run': run_id, 'current': curr, 'fwhm': fwhm85, 'err': err85,
        'amp': popt85[2] * 1e3, 'res_rms': np.std(res85) * 1e3
    })
    data_85[curr] = (x85, y85, popt85, res85, fwhm85, err85, run_id)

df_87 = pd.DataFrame(results_87).sort_values('current')
df_85 = pd.DataFrame(results_85).sort_values('current')

# =========================================================================
# FIGURE 1: Summary of Linewidth vs Current & Amplitude vs Current (14 Points)
# =========================================================================
fig1, (ax1a, ax1b, ax1c) = plt.subplots(1, 3, figsize=(18, 5.5))

# Parabolic fits on consecutive runs (9 to 69 mA)
c_sub = df_87[df_87['current'] > 0]['current']
p_87 = np.polyfit(c_sub, df_87[df_87['current'] > 0]['fwhm'], 2)
p_85 = np.polyfit(c_sub, df_85[df_85['current'] > 0]['fwhm'], 2)
c_dense = np.linspace(0, 70, 200)

ax1a.errorbar(df_87['current'], df_87['fwhm'], yerr=df_87['err'], fmt='o', color='#1f77b4', ms=6, capsize=4, label='$^{87}$Rb Data (14 points)')
ax1a.plot(c_dense, np.polyval(p_87, c_dense), '-', color='#1f77b4', lw=1.8, label=f'$^{{87}}$Rb Parabola (Min = {np.min(np.polyval(p_87, c_dense)):.2f} MHz at {(-p_87[1]/(2*p_87[0])):.1f} mA)')

ax1a.errorbar(df_85['current'], df_85['fwhm'], yerr=df_85['err'], fmt='s', color='#ff7f0e', ms=6, capsize=4, label='$^{85}$Rb Data (14 points)')
ax1a.plot(c_dense, np.polyval(p_85, c_dense), '--', color='#ff7f0e', lw=1.8, label=f'$^{{85}}$Rb Parabola (Min = {np.min(np.polyval(p_85, c_dense)):.2f} MHz at {(-p_85[1]/(2*p_85[0])):.1f} mA)')

ax1a.axhline(6.07, color='black', ls=':', label='Natural Linewidth (6.07 MHz)')
ax1a.axvspan(16, 30, color='green', alpha=0.1, label='Cancellation Window (16–30 mA)')
ax1a.set_xlabel('Coil Current (mA)', fontsize=11)
ax1a.set_ylabel('FWHM Linewidth (MHz)', fontsize=11)
ax1a.set_title('(a) Linewidth vs Current: 14 Datasets Mapped', fontweight='bold', fontsize=12)
ax1a.grid(True, alpha=0.3)
ax1a.legend(fontsize=8.5, loc='upper right')

# Amplitude vs Current
ax1b.plot(df_87['current'], df_87['amp'], 'o-', color='#1f77b4', lw=1.5, ms=6, label='$^{87}$Rb ($F=2 \\to F\'=3$)')
ax1b.plot(df_85['current'], df_85['amp'], 's--', color='#ff7f0e', lw=1.5, ms=6, label='$^{85}$Rb ($F=3 \\to F\'=4$)')
ax1b.set_xlabel('Coil Current (mA)', fontsize=11)
ax1b.set_ylabel('Peak Amplitude (mV)', fontsize=11)
ax1b.set_title('(b) Peak Amplitude vs Current', fontweight='bold', fontsize=12)
ax1b.grid(True, alpha=0.3)
ax1b.legend(fontsize=10)

# Residual RMS vs Current
ax1c.plot(df_87['current'], df_87['res_rms'], 'o-', color='#1f77b4', lw=1.5, ms=6, label='$^{87}$Rb Fit RMS Residual')
ax1c.plot(df_85['current'], df_85['res_rms'], 's--', color='#ff7f0e', lw=1.5, ms=6, label='$^{85}$Rb Fit RMS Residual')
ax1c.set_xlabel('Coil Current (mA)', fontsize=11)
ax1c.set_ylabel('Fit Residual RMS (mV)', fontsize=11)
ax1c.set_title('(c) Fit Residual vs Field (Smallest at Cancellation)', fontweight='bold', fontsize=12)
ax1c.grid(True, alpha=0.3)
ax1c.legend(fontsize=10)

plt.tight_layout()
fig1_path = os.path.join(out_dir, 'session10_linewidth_vs_current_14runs.png')
fig1.savefig(fig1_path, dpi=160)
plt.close(fig1)
print(f'Saved Figure 1 to {fig1_path}')

# =========================================================================
# FIGURE 2: 87Rb Cycling Transition - 14 Separate Dedicated Plots (Sorted by Current)
# =========================================================================
sorted_currents = sorted(currents_map.values())
fig2, axes2 = plt.subplots(7, 2, figsize=(15, 26))
axes2 = axes2.flatten()

for idx, curr in enumerate(sorted_currents):
    ax = axes2[idx]
    x_dat, y_dat, popt, res, fwhm, err, r_id = data_87[curr]
    x_dense = np.linspace(-25, 25, 300)
    y_fit = lorentzian(x_dense, *popt)
    
    ax.plot(x_dat, y_dat * 1e3, '.', color='#1f77b4', alpha=0.35, ms=3.5, label='Data')
    ax.plot(x_dense, y_fit * 1e3, 'r-', lw=1.8, label=f'Fit (FWHM = {fwhm:.2f} $\\pm$ {err:.2f} MHz)')
    ax.plot(x_dense, (popt[3] + popt[4] * (x_dense - popt[0])) * 1e3, 'k:', lw=1, label='Baseline')
    
    ax.set_title(f'Run {r_id} ({curr} mA) — $^{{87}}$Rb ($F=2 \\to F\'=3$): FWHM = {fwhm:.2f} MHz', fontweight='bold', fontsize=10.5)
    ax.set_xlabel('Relative Frequency (MHz)', fontsize=9.5)
    ax.set_ylabel('Signal (mV)', fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    res_rms = np.std(res) * 1e3
    ax.text(0.03, 0.72, f'Peak: {popt[2]*1e3:.1f} mV\nRMS: {res_rms:.3f} mV', 
            transform=ax.transAxes, fontsize=8, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8))

plt.tight_layout()
fig2_path = os.path.join(out_dir, 'session10_87rb_all_14_fits.png')
fig2.savefig(fig2_path, dpi=150)
plt.close(fig2)
print(f'Saved Figure 2 to {fig2_path}')

# =========================================================================
# FIGURE 3: 85Rb Cycling Transition - 14 Separate Dedicated Plots (Sorted by Current)
# =========================================================================
fig3, axes3 = plt.subplots(7, 2, figsize=(15, 26))
axes3 = axes3.flatten()

for idx, curr in enumerate(sorted_currents):
    ax = axes3[idx]
    x_dat, y_dat, popt, res, fwhm, err, r_id = data_85[curr]
    x_dense = np.linspace(-20, 20, 300)
    y_fit = lorentzian(x_dense, *popt)
    
    ax.plot(x_dat, y_dat * 1e3, '.', color='#ff7f0e', alpha=0.35, ms=3.5, label='Data')
    ax.plot(x_dense, y_fit * 1e3, 'r-', lw=1.8, label=f'Fit (FWHM = {fwhm:.2f} $\\pm$ {err:.2f} MHz)')
    ax.plot(x_dense, (popt[3] + popt[4] * (x_dense - popt[0])) * 1e3, 'k:', lw=1, label='Baseline')
    
    ax.set_title(f'Run {r_id} ({curr} mA) — $^{{85}}$Rb ($F=3 \\to F\'=4$): FWHM = {fwhm:.2f} MHz', fontweight='bold', fontsize=10.5)
    ax.set_xlabel('Relative Frequency (MHz)', fontsize=9.5)
    ax.set_ylabel('Signal (mV)', fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    res_rms = np.std(res) * 1e3
    ax.text(0.03, 0.72, f'Peak: {popt[2]*1e3:.1f} mV\nRMS: {res_rms:.3f} mV', 
            transform=ax.transAxes, fontsize=8, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8))

plt.tight_layout()
fig3_path = os.path.join(out_dir, 'session10_85rb_all_14_fits.png')
fig3.savefig(fig3_path, dpi=150)
plt.close(fig3)
print(f'Saved Figure 3 to {fig3_path}')

print('All 14-run figures generated successfully!')
