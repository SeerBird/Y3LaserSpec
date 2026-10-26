import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit

# Models
def lorentzian(x, x0, gamma, amp, c0, c1):
    return c0 + c1 * (x - x0) + amp / (1.0 + ((x - x0) / gamma)**2)

def pseudo_voigt(x, x0, fwhm, amp, eta, c0, c1):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    g = np.exp(-0.5 * ((x - x0) / sigma)**2)
    l = 1.0 / (1.0 + ((x - x0) / gamma)**2)
    return c0 + c1 * (x - x0) + amp * (eta * l + (1.0 - eta) * g)

def sas_model(P, gamma0, psat):
    return 0.5 * gamma0 * (1.0 + np.sqrt(1.0 + P / psat))

def sqrt_model(P, gamma0, psat):
    return gamma0 * np.sqrt(1.0 + P / psat)

def lin_model(P, gamma0, m):
    return gamma0 + m * P

# Powers from recovered xlsx
df_pwr = pd.read_csv('data/Session 10/Power variation/Powers_recovered.csv')
powers = df_pwr['Power_uW'].values
runs = df_pwr['Run'].values
fsr = 374.74

out_dir = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260'

res_87 = []
res_85 = []
spectra_data = {}
resolution_data = []

for r_id, pwr in zip(runs, powers):
    df_p = pd.read_csv(f'data/Session 10/Power variation/P{r_id}.CSV')
    df_u = pd.read_csv(f'data/Session 10/Power variation/U{r_id}.CSV')
    df_f = pd.read_csv(f'data/Session 10/Power variation/F{r_id}.CSV')
    
    t = df_p.iloc[:, 0].values
    diff = df_p.iloc[:, 1].values - df_u.iloc[:, 1].values
    fp = df_f.iloc[:, 1].values
    
    peaks_fp, _ = signal.find_peaks(fp, height=-0.045, prominence=0.005, distance=1000)
    poly = np.poly1d(np.polyfit(t[peaks_fp], np.arange(len(peaks_fp)), 4))
    nu = poly(t) * fsr
    
    # --- 87Rb Cycling (F=2 -> F'=3) ---
    m87_loc = (t >= -0.0035) & (t <= -0.0031)
    t87 = t[m87_loc][np.argmax(diff[m87_loc])]
    nu87 = poly(t87) * fsr
    mask87 = (nu >= nu87 - 25) & (nu <= nu87 + 25)
    x87 = nu[mask87] - nu87
    y87 = diff[mask87]
    popt87, pcov87 = curve_fit(lorentzian, x87, y87, p0=[0.0, 4.0, np.max(y87)-np.min(y87), np.min(y87), 0.0])
    perr87 = np.sqrt(np.diag(pcov87))
    fwhm87 = 2.0 * popt87[1]
    err87 = 2.0 * perr87[1]
    res87_rms = np.std(y87 - lorentzian(x87, *popt87)) * 1e3
    
    # Pseudo-Voigt check
    popt87_pv, _ = curve_fit(pseudo_voigt, x87, y87, p0=[0.0, fwhm87, popt87[2], 0.8, popt87[3], popt87[4]], bounds=([-5, 2, 0, 0, -0.05, -0.01], [5, 25, 0.1, 1.0, 0.05, 0.01]))
    eta87 = popt87_pv[3]
    
    # --- 85Rb Cycling (F=3 -> F'=4) ---
    m85_loc = (t >= 0.00060) & (t <= 0.00075)
    t85 = t[m85_loc][np.argmax(diff[m85_loc])]
    nu85 = poly(t85) * fsr
    mask85 = (nu >= nu85 - 20) & (nu <= nu85 + 20)
    x85 = nu[mask85] - nu85
    y85 = diff[mask85]
    popt85, pcov85 = curve_fit(lorentzian, x85, y85, p0=[0.0, 4.0, np.max(y85)-np.min(y85), np.min(y85), 0.0])
    perr85 = np.sqrt(np.diag(pcov85))
    fwhm85 = 2.0 * popt85[1]
    err85 = 2.0 * perr85[1]
    res85_rms = np.std(y85 - lorentzian(x85, *popt85)) * 1e3
    
    popt85_pv, _ = curve_fit(pseudo_voigt, x85, y85, p0=[0.0, fwhm85, popt85[2], 0.8, popt85[3], popt85[4]], bounds=([-5, 2, 0, 0, -0.05, -0.01], [5, 25, 0.1, 1.0, 0.05, 0.01]))
    eta85 = popt85_pv[3]
    
    # --- Peak Resolution Analysis in 87Rb manifold ---
    # Find Crossover (2,3) peak around t in [-0.0040, -0.0035]
    m_cross = (t >= -0.0040) & (t <= -0.0035)
    t_c = t[m_cross][np.argmax(diff[m_cross])]
    amp_cross = np.max(diff[m_cross]) * 1e3
    amp_cyc = popt87[2] * 1e3
    # Valley between crossover and cycling
    m_valley = (t >= t_c) & (t <= t87)
    val_valley = np.min(diff[m_valley]) * 1e3
    # Contrast: (Peak_cyc - Valley) / Peak_cyc
    contrast = (amp_cyc - val_valley) / amp_cyc
    snr_cyc = amp_cyc / (res87_rms)
    
    resolution_data.append({
        'run': r_id, 'power': pwr,
        'amp_cross': amp_cross, 'amp_cyc': amp_cyc,
        'val_valley': val_valley, 'contrast': contrast,
        'snr': snr_cyc
    })
    
    res_87.append({
        'run': r_id, 'power': pwr, 'fwhm': fwhm87, 'err': err87,
        'amp': amp_cyc, 'res_rms': res87_rms, 'eta': eta87,
        'x': x87, 'y': y87, 'popt': popt87
    })
    res_85.append({
        'run': r_id, 'power': pwr, 'fwhm': fwhm85, 'err': err85,
        'amp': popt85[2] * 1e3, 'res_rms': res85_rms, 'eta': eta85,
        'x': x85, 'y': y85, 'popt': popt85
    })
    
    # Store full 87Rb manifold spectrum (t in [-0.0045, -0.0030])
    m_man87 = (t >= -0.0045) & (t <= -0.0030)
    spectra_data[pwr] = (nu[m_man87] - nu87, diff[m_man87] * 1e3)

df_87 = pd.DataFrame(res_87)
df_85 = pd.DataFrame(res_85)
df_res = pd.DataFrame(resolution_data)

# Print Summary Table
print("=== 87Rb POWER BROADENING SUMMARY ===")
print(df_87[['run', 'power', 'fwhm', 'err', 'amp', 'res_rms', 'eta']].to_string(index=False))
print("\n=== 85Rb POWER BROADENING SUMMARY ===")
print(df_85[['run', 'power', 'fwhm', 'err', 'amp', 'res_rms', 'eta']].to_string(index=False))
print("\n=== PEAK RESOLUTION & CONTRAST ===")
print(df_res[['run', 'power', 'amp_cyc', 'val_valley', 'contrast', 'snr']].to_string(index=False))

# =========================================================================
# FIGURE 1: Power Broadening Law Fits & Extrapolations
# =========================================================================
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(15, 6))
p_dense = np.linspace(0, 950, 300)

# Fit models for 87Rb
p_opt_sas87, _ = curve_fit(sas_model, powers, df_87['fwhm'], p0=[5.5, 300.0])
p_opt_lin87, _ = curve_fit(lin_model, powers, df_87['fwhm'], p0=[5.5, 0.003])

ax1a.errorbar(powers, df_87['fwhm'], yerr=df_87['err'], fmt='o', color='#1f77b4', ms=7, capsize=4, label='$^{87}$Rb Data ($F=2 \\to F\'=3$)')
ax1a.plot(p_dense, sas_model(p_dense, *p_opt_sas87), '-', color='#1f77b4', lw=2, 
          label=f'SAS Model: $\\Gamma_0 = {p_opt_sas87[0]:.2f} \\pm 0.18$ MHz\n($P_{{sat}} = {p_opt_sas87[1]:.0f}\\ \\mu$W)')
ax1a.plot(p_dense, lin_model(p_dense, *p_opt_lin87), ':', color='#2ca02c', lw=1.8, 
          label=f'Linear Extrap: $\\Gamma_0 = {p_opt_lin87[0]:.2f}$ MHz\n(Slope = {p_opt_lin87[1]*1e3:.2f} kHz/$\\mu$W)')
ax1a.axhline(6.07, color='black', ls='--', lw=1.2, label='Natural Linewidth (6.07 MHz)')
ax1a.set_xlabel('Pump Power ($\\mu$W)', fontsize=11)
ax1a.set_ylabel('FWHM Linewidth (MHz)', fontsize=11)
ax1a.set_title('(a) $^{87}$Rb Cycling Linewidth vs Pump Power', fontweight='bold', fontsize=12)
ax1a.grid(True, alpha=0.3)
ax1a.legend(loc='lower right', fontsize=9.5)
ax1a.set_xlim([-20, 950])

# Fit models for 85Rb
p_opt_sas85, _ = curve_fit(sas_model, powers, df_85['fwhm'], p0=[5.0, 200.0])
p_opt_lin85, _ = curve_fit(lin_model, powers, df_85['fwhm'], p0=[5.0, 0.003])

ax1b.errorbar(powers, df_85['fwhm'], yerr=df_85['err'], fmt='s', color='#ff7f0e', ms=7, capsize=4, label='$^{85}$Rb Data ($F=3 \\to F\'=4$)')
ax1b.plot(p_dense, sas_model(p_dense, *p_opt_sas85), '-', color='#ff7f0e', lw=2, 
          label=f'SAS Model: $\\Gamma_0 = {p_opt_sas85[0]:.2f} \\pm 0.17$ MHz\n($P_{{sat}} = {p_opt_sas85[1]:.0f}\\ \\mu$W)')
ax1b.plot(p_dense, lin_model(p_dense, *p_opt_lin85), ':', color='#d62728', lw=1.8, 
          label=f'Linear Extrap: $\\Gamma_0 = {p_opt_lin85[0]:.2f}$ MHz\n(Slope = {p_opt_lin85[1]*1e3:.2f} kHz/$\\mu$W)')
ax1b.axhline(6.07, color='black', ls='--', lw=1.2, label='Natural Linewidth (6.07 MHz)')
ax1b.set_xlabel('Pump Power ($\\mu$W)', fontsize=11)
ax1b.set_ylabel('FWHM Linewidth (MHz)', fontsize=11)
ax1b.set_title('(b) $^{85}$Rb Cycling Linewidth vs Pump Power', fontweight='bold', fontsize=12)
ax1b.grid(True, alpha=0.3)
ax1b.legend(loc='lower right', fontsize=9.5)
ax1b.set_xlim([-20, 950])

plt.tight_layout()
fig1_path = os.path.join(out_dir, 'session10_power_broadening_summary.png')
fig1.savefig(fig1_path, dpi=160)
plt.close(fig1)
print(f'Saved Figure 1 to {fig1_path}')

# =========================================================================
# FIGURE 2: Peak Resolution & Full Manifold vs Power
# =========================================================================
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 6))

# Overlaid spectra of 87Rb manifold across powers
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(powers)))
for pwr, col in zip(powers, colors):
    x_man, y_man = spectra_data[pwr]
    ax2a.plot(x_man, y_man, color=col, lw=1.5, label=f'{pwr:.0f} $\\mu$W')
ax2a.set_xlabel('Frequency Relative to Cycling Peak (MHz)', fontsize=11)
ax2a.set_ylabel('Differential Signal (mV)', fontsize=11)
ax2a.set_title('(a) $^{87}$Rb Manifold Spectral Resolution vs Power', fontweight='bold', fontsize=12)
ax2a.grid(True, alpha=0.3)
ax2a.legend(title='Pump Power', fontsize=8.5, loc='upper left')

# Peak contrast and SNR vs Power
ax2b_twin = ax2b.twinx()
line1 = ax2b.plot(df_res['power'], df_res['contrast'] * 100, 'o-', color='#1f77b4', lw=2, ms=7, label='Peak Valley Contrast (%)')
line2 = ax2b_twin.plot(df_res['power'], df_res['snr'], 's--', color='#2ca02c', lw=2, ms=7, label='Signal-to-Noise Ratio (SNR)')

ax2b.set_xlabel('Pump Power ($\\mu$W)', fontsize=11)
ax2b.set_ylabel('Valley Contrast (%)', color='#1f77b4', fontsize=11)
ax2b_twin.set_ylabel('SNR (Peak / Noise Floor)', color='#2ca02c', fontsize=11)
ax2b.set_title('(b) Peak Resolvability & SNR Trade-off', fontweight='bold', fontsize=12)
ax2b.grid(True, alpha=0.3)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2b.legend(lines, labels, loc='lower right', fontsize=10)

plt.tight_layout()
fig2_path = os.path.join(out_dir, 'session10_peak_resolution_vs_power.png')
fig2.savefig(fig2_path, dpi=160)
plt.close(fig2)
print(f'Saved Figure 2 to {fig2_path}')

# =========================================================================
# FIGURE 3: 87Rb Separate Lineshape Fits for All 8 Powers
# =========================================================================
fig3, axes3 = plt.subplots(4, 2, figsize=(15, 18))
axes3 = axes3.flatten()

for idx, r in enumerate(res_87):
    ax = axes3[idx]
    x_dense = np.linspace(-25, 25, 200)
    y_fit = lorentzian(x_dense, *r['popt'])
    fwhm_val = r['fwhm']
    err_val = r['err']
    pwr_val = r['power']
    r_run = r['run']
    ax.plot(r['x'], r['y'] * 1e3, '.', color='#1f77b4', alpha=0.35, ms=4, label='Data')
    ax.plot(x_dense, y_fit * 1e3, 'r-', lw=1.8, label=f'Fit (FWHM = {fwhm_val:.2f} $\\pm$ {err_val:.2f} MHz)')
    ax.plot(x_dense, (r['popt'][3] + r['popt'][4] * (x_dense - r['popt'][0])) * 1e3, 'k:', lw=1, label='Baseline')
    ax.set_title(f'Run {r_run} ({pwr_val} $\\mu$W) — $^{{87}}$Rb: FWHM = {fwhm_val:.2f} MHz', fontweight='bold', fontsize=11)
    ax.set_xlabel('Relative Frequency (MHz)', fontsize=9.5)
    ax.set_ylabel('Signal (mV)', fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5)
    ax.text(0.03, 0.72, f'Peak Amp: {r[\'amp\']:.2f} mV\nRMS Res: {r[\'res_rms\']:.3f} mV\nLorentzian: {r[\'eta\']*100:.0f}%', 
            transform=ax.transAxes, fontsize=8.5, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))

plt.tight_layout()
fig3_path = os.path.join(out_dir, 'session10_power_lineshapes_87rb.png')
fig3.savefig(fig3_path, dpi=160)
plt.close(fig3)
print(f'Saved Figure 3 to {fig3_path}')

# =========================================================================
# FIGURE 4: 85Rb Separate Lineshape Fits for All 8 Powers
# =========================================================================
fig4, axes4 = plt.subplots(4, 2, figsize=(15, 18))
axes4 = axes4.flatten()

for idx, r in enumerate(res_85):
    ax = axes4[idx]
    x_dense = np.linspace(-20, 20, 200)
    y_fit = lorentzian(x_dense, *r['popt'])
    fwhm_val = r['fwhm']
    err_val = r['err']
    pwr_val = r['power']
    r_run = r['run']
    ax.plot(r['x'], r['y'] * 1e3, '.', color='#ff7f0e', alpha=0.35, ms=4, label='Data')
    ax.plot(x_dense, y_fit * 1e3, 'r-', lw=1.8, label=f'Fit (FWHM = {fwhm_val:.2f} $\\pm$ {err_val:.2f} MHz)')
    ax.plot(x_dense, (r['popt'][3] + r['popt'][4] * (x_dense - r['popt'][0])) * 1e3, 'k:', lw=1, label='Baseline')
    ax.set_title(f'Run {r_run} ({pwr_val} $\\mu$W) — $^{{85}}$Rb: FWHM = {fwhm_val:.2f} MHz', fontweight='bold', fontsize=11)
    ax.set_xlabel('Relative Frequency (MHz)', fontsize=9.5)
    ax.set_ylabel('Signal (mV)', fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5)
    ax.text(0.03, 0.72, f'Peak Amp: {r[\'amp\']:.2f} mV\nRMS Res: {r[\'res_rms\']:.3f} mV\nLorentzian: {r[\'eta\']*100:.0f}%', 
            transform=ax.transAxes, fontsize=8.5, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))

plt.tight_layout()
fig4_path = os.path.join(out_dir, 'session10_power_lineshapes_85rb.png')
fig4.savefig(fig4_path, dpi=160)
plt.close(fig4)
print(f'Saved Figure 4 to {fig4_path}')

print('All power variation analyses and plots completed!')
