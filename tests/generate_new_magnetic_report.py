import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit

# Define Lorentzian model
def lorentzian(x, x0, gamma, amp, c0, c1):
    return c0 + c1 * (x - x0) + amp / (1.0 + ((x - x0) / gamma)**2)

currents = [0, 9, 19, 29, 39, 48.5, 59, 69]
fsr = 374.74  # MHz

results_87 = []
results_85 = []

# Prepare data storage for plotting
data_87 = {}
data_85 = {}

out_dir = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260'
fits_dir = os.path.join(out_dir, 'session10_fits')
os.makedirs(fits_dir, exist_ok=True)

for run_id, curr in enumerate(currents, 1):
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
    data_87[curr] = (x87, y87, popt87, res87, fwhm87, err87)
    
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
    data_85[curr] = (x85, y85, popt85, res85, fwhm85, err85)

# Convert to DataFrame
df_87 = pd.DataFrame(results_87)
df_85 = pd.DataFrame(results_85)

# =========================================================================
# FIGURE 1: Summary of Linewidth vs Current & Amplitude vs Current
# =========================================================================
fig1, (ax1a, ax1b, ax1c) = plt.subplots(1, 3, figsize=(18, 5.5))

# Parabolic fits
p_87 = np.polyfit(df_87['current'], df_87['fwhm'], 2)
p_85 = np.polyfit(df_85['current'], df_85['fwhm'], 2)
c_dense = np.linspace(0, 70, 200)

ax1a.errorbar(df_87['current'], df_87['fwhm'], yerr=df_87['err'], fmt='o', color='#1f77b4', ms=7, capsize=4, label='$^{87}$Rb ($F=2 \\to F\'=3$) Data')
ax1a.plot(c_dense, np.polyval(p_87, c_dense), '-', color='#1f77b4', lw=1.8, label=f'$^{{87}}$Rb Parabola (Min = {np.min(np.polyval(p_87, c_dense)):.2f} MHz)')

ax1a.errorbar(df_85['current'], df_85['fwhm'], yerr=df_85['err'], fmt='s', color='#ff7f0e', ms=7, capsize=4, label='$^{85}$Rb ($F=3 \\to F\'=4$) Data')
ax1a.plot(c_dense, np.polyval(p_85, c_dense), '--', color='#ff7f0e', lw=1.8, label=f'$^{{85}}$Rb Parabola (Min = {np.min(np.polyval(p_85, c_dense)):.2f} MHz)')

ax1a.axhline(6.07, color='black', ls=':', label='Natural Linewidth (6.07 MHz)')
ax1a.set_xlabel('Coil Current (mA)', fontsize=11)
ax1a.set_ylabel('FWHM Linewidth (MHz)', fontsize=11)
ax1a.set_title('(a) Linewidth vs Current: Clear Cancellation Minimum', fontweight='bold', fontsize=12)
ax1a.grid(True, alpha=0.3)
ax1a.legend(fontsize=9, loc='upper right')

# Amplitude vs Current
ax1b.plot(df_87['current'], df_87['amp'], 'o-', color='#1f77b4', lw=1.8, ms=7, label='$^{87}$Rb ($F=2 \\to F\'=3$)')
ax1b.plot(df_85['current'], df_85['amp'], 's--', color='#ff7f0e', lw=1.8, ms=7, label='$^{85}$Rb ($F=3 \\to F\'=4$)')
ax1b.set_xlabel('Coil Current (mA)', fontsize=11)
ax1b.set_ylabel('Peak Amplitude (mV)', fontsize=11)
ax1b.set_title('(b) Peak Amplitude vs Current', fontweight='bold', fontsize=12)
ax1b.grid(True, alpha=0.3)
ax1b.legend(fontsize=10)

# Residual RMS vs Current
ax1c.plot(df_87['current'], df_87['res_rms'], 'o-', color='#1f77b4', lw=1.8, ms=7, label='$^{87}$Rb Fit RMS Residual')
ax1c.plot(df_85['current'], df_85['res_rms'], 's--', color='#ff7f0e', lw=1.8, ms=7, label='$^{85}$Rb Fit RMS Residual')
ax1c.set_xlabel('Coil Current (mA)', fontsize=11)
ax1c.set_ylabel('Fit Residual RMS (mV)', fontsize=11)
ax1c.set_title('(c) Fit Residual vs Field (Zeeman Distortion at High B)', fontweight='bold', fontsize=12)
ax1c.grid(True, alpha=0.3)
ax1c.legend(fontsize=10)

plt.tight_layout()
fig1_path = os.path.join(out_dir, 'session10_linewidth_vs_current_summary.png')
fig1.savefig(fig1_path, dpi=160)
plt.close(fig1)
print(f'Saved Figure 1 to {fig1_path}')

# =========================================================================
# FIGURE 2: 87Rb Cycling Transition - 8 Separate Dedicated Plots
# =========================================================================
fig2, axes2 = plt.subplots(4, 2, figsize=(15, 18))
axes2 = axes2.flatten()

for idx, curr in enumerate(currents):
    ax = axes2[idx]
    x_dat, y_dat, popt, res, fwhm, err = data_87[curr]
    
    # 2-panel inset style: Top = fit, Bottom = residual
    # Use GridSpec inside or divide the subplot
    x_dense = np.linspace(-25, 25, 300)
    y_fit = lorentzian(x_dense, *popt)
    
    ax.plot(x_dat, y_dat * 1e3, '.', color='#1f77b4', alpha=0.35, ms=4, label='Data')
    ax.plot(x_dense, y_fit * 1e3, 'r-', lw=1.8, label=f'Lorentzian Fit (FWHM = {fwhm:.2f} $\\pm$ {err:.2f} MHz)')
    
    # Show baseline
    ax.plot(x_dense, (popt[3] + popt[4] * (x_dense - popt[0])) * 1e3, 'k:', lw=1, label='Linear Baseline')
    
    ax.set_title(f'Run {idx+1} ({curr} mA) — $^{{87}}$Rb ($F=2 \\to F\'=3$): FWHM = {fwhm:.2f} MHz', fontweight='bold', fontsize=11)
    ax.set_xlabel('Relative Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Signal (mV)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5)
    
    # Add text box with fit stats
    res_rms = np.std(res) * 1e3
    ax.text(0.03, 0.72, f'Peak Amp: {popt[2]*1e3:.1f} mV\nRMS Res: {res_rms:.3f} mV', 
            transform=ax.transAxes, fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
fig2_path = os.path.join(out_dir, 'session10_87rb_separate_fits.png')
fig2.savefig(fig2_path, dpi=160)
plt.close(fig2)
print(f'Saved Figure 2 to {fig2_path}')

# =========================================================================
# FIGURE 3: 85Rb Cycling Transition - 8 Separate Dedicated Plots
# =========================================================================
fig3, axes3 = plt.subplots(4, 2, figsize=(15, 18))
axes3 = axes3.flatten()

for idx, curr in enumerate(currents):
    ax = axes3[idx]
    x_dat, y_dat, popt, res, fwhm, err = data_85[curr]
    
    x_dense = np.linspace(-20, 20, 300)
    y_fit = lorentzian(x_dense, *popt)
    
    ax.plot(x_dat, y_dat * 1e3, '.', color='#ff7f0e', alpha=0.35, ms=4, label='Data')
    ax.plot(x_dense, y_fit * 1e3, 'r-', lw=1.8, label=f'Lorentzian Fit (FWHM = {fwhm:.2f} $\\pm$ {err:.2f} MHz)')
    ax.plot(x_dense, (popt[3] + popt[4] * (x_dense - popt[0])) * 1e3, 'k:', lw=1, label='Linear Baseline')
    
    ax.set_title(f'Run {idx+1} ({curr} mA) — $^{{85}}$Rb ($F=3 \\to F\'=4$): FWHM = {fwhm:.2f} MHz', fontweight='bold', fontsize=11)
    ax.set_xlabel('Relative Frequency (MHz)', fontsize=10)
    ax.set_ylabel('Signal (mV)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5)
    
    res_rms = np.std(res) * 1e3
    ax.text(0.03, 0.72, f'Peak Amp: {popt[2]*1e3:.1f} mV\nRMS Res: {res_rms:.3f} mV', 
            transform=ax.transAxes, fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
fig3_path = os.path.join(out_dir, 'session10_85rb_separate_fits.png')
fig3.savefig(fig3_path, dpi=160)
plt.close(fig3)
print(f'Saved Figure 3 to {fig3_path}')

# =========================================================================
# FIGURE 4: Individual Standalone Detailed Fits (with Inset Residuals)
# =========================================================================
for idx, curr in enumerate(currents):
    # For 87Rb
    f, (a_top, a_bot) = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    x_dat, y_dat, popt, res, fwhm, err = data_87[curr]
    x_dense = np.linspace(-25, 25, 300)
    a_top.plot(x_dat, y_dat * 1e3, '.', color='#1f77b4', alpha=0.4, ms=4, label='Data')
    a_top.plot(x_dense, lorentzian(x_dense, *popt) * 1e3, 'r-', lw=1.8, label=f'Fit (FWHM={fwhm:.2f} MHz)')
    a_top.set_ylabel('Signal (mV)')
    a_top.set_title(f'$^{{87}}$Rb ($F=2 \\to F\'=3$) — Run {idx+1} ({curr} mA)', fontweight='bold')
    a_top.grid(True, alpha=0.3)
    a_top.legend(loc='upper right')
    
    a_bot.plot(x_dat, res * 1e3, '.', color='#1f77b4', ms=3, alpha=0.5)
    a_bot.axhline(0, color='black', lw=0.8, ls=':')
    a_bot.set_ylabel('Res (mV)')
    a_bot.set_xlabel('Relative Frequency (MHz)')
    a_bot.set_ylim([-1.0, 1.0])
    a_bot.grid(True, alpha=0.3)
    plt.tight_layout()
    f.savefig(os.path.join(fits_dir, f'fit_87rb_run{idx+1}_{curr}ma.png'), dpi=120)
    plt.close(f)

print('All figures successfully generated and saved!')
