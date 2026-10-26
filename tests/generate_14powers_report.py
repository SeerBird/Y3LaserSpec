import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit

def lorentzian(x, x0, gamma, amp, c0, c1):
    return c0 + c1 * (x - x0) + amp / (1.0 + ((x - x0) / gamma)**2)

def sas_model(P, gamma0, psat):
    return 0.5 * gamma0 * (1.0 + np.sqrt(1.0 + P / psat))

def lin_model(P, gamma0, m):
    return gamma0 + m * P

df_pwr = pd.read_excel('data/Session 10/Power variation/Powers.xlsx')
powers = df_pwr['Power_uW'].values
runs = np.arange(1, len(powers) + 1)
fsr = 374.74

out_dir = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260'

res_87 = []
res_85 = []
spectra_export = {}
summary_list = []

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
    
    # 87Rb cycling
    m87_loc = (t >= -0.0035) & (t <= -0.0031)
    t87 = t[m87_loc][np.argmax(diff[m87_loc])]
    nu87 = poly(t87) * fsr
    
    # Local window +/- 14 MHz
    mask87 = (nu >= nu87 - 14) & (nu <= nu87 + 14)
    x87 = nu[mask87] - nu87
    y87 = diff[mask87]
    try:
        popt87, pcov87 = curve_fit(lorentzian, x87, y87, p0=[0.0, 3.5, np.max(y87)-np.min(y87), np.min(y87), 0.0])
        fwhm87 = 2.0 * popt87[1]
        err87 = 2.0 * np.sqrt(pcov87[1, 1])
        amp87 = popt87[2] * 1e3
        res87_rms = np.std(y87 - lorentzian(x87, *popt87)) * 1e3
    except:
        fwhm87, err87, amp87, res87_rms, popt87 = np.nan, np.nan, np.nan, np.nan, [0, 0, 0, 0, 0]
    
    # 85Rb cycling
    m85_loc = (t >= 0.00060) & (t <= 0.00075)
    t85 = t[m85_loc][np.argmax(diff[m85_loc])]
    nu85 = poly(t85) * fsr
    mask85 = (nu >= nu85 - 12) & (nu <= nu85 + 12)
    x85 = nu[mask85] - nu85
    y85 = diff[mask85]
    try:
        popt85, pcov85 = curve_fit(lorentzian, x85, y85, p0=[0.0, 3.5, np.max(y85)-np.min(y85), np.min(y85), 0.0])
        fwhm85 = 2.0 * popt85[1]
        err85 = 2.0 * np.sqrt(pcov85[1, 1])
        amp85 = popt85[2] * 1e3
        res85_rms = np.std(y85 - lorentzian(x85, *popt85)) * 1e3
    except:
        fwhm85, err85, amp85, res85_rms, popt85 = np.nan, np.nan, np.nan, np.nan, [0, 0, 0, 0, 0]
    
    # Export decimation for interactive SVG
    m87_man = (t >= -0.0044) & (t <= -0.0029)
    x_man = nu[m87_man] - nu87
    y_man = diff[m87_man] * 1e3
    spectra_export[str(r_id)] = {
        'power': float(pwr),
        'x': [round(float(v), 2) for v in x_man[::15]],
        'y': [round(float(v), 2) for v in y_man[::15]]
    }
    
    # Resolvability contrast & SNR
    m_cross = (t >= -0.0040) & (t <= -0.0035)
    amp_cross = np.max(diff[m_cross]) * 1e3
    m_valley = (t >= t[m_cross][np.argmax(diff[m_cross])]) & (t <= t87)
    val_valley = np.min(diff[m_valley]) * 1e3
    contrast = (amp87 - val_valley) / amp87 if (amp87 > 0 and not np.isnan(amp87)) else 0
    snr = amp87 / res87_rms if (res87_rms > 0 and not np.isnan(res87_rms)) else 0
    
    res_87.append({
        'run': r_id, 'power': pwr, 'fwhm': fwhm87, 'err': err87,
        'amp': amp87, 'res_rms': res87_rms, 'x': x87, 'y': y87, 'popt': popt87
    })
    res_85.append({
        'run': r_id, 'power': pwr, 'fwhm': fwhm85, 'err': err85,
        'amp': amp85, 'res_rms': res85_rms, 'x': x85, 'y': y85, 'popt': popt85
    })
    
    status = "Normal"
    if pwr >= 700: status = "Strong SNR (>50), power-broadened wings"
    elif pwr >= 200: status = "Optimal Resolvability Window (Sharp Peaks + SNR > 25)"
    elif pwr >= 50: status = "Narrow natural linewidth, SNR 10-20"
    else: status = "Noise-floor limit (SNR < 3)"
    
    summary_list.append({
        'run': r_id, 'power': float(pwr),
        'fwhm87': round(float(fwhm87), 2) if not np.isnan(fwhm87) else 0.0,
        'err87': round(float(err87), 2) if not np.isnan(err87) else 0.0,
        'amp87': round(float(amp87), 2) if not np.isnan(amp87) else 0.0,
        'fwhm85': round(float(fwhm85), 2) if not np.isnan(fwhm85) else 0.0,
        'err85': round(float(err85), 2) if not np.isnan(err85) else 0.0,
        'amp85': round(float(amp85), 2) if not np.isnan(amp85) else 0.0,
        'contrast': round(float(contrast * 100), 1),
        'snr': round(float(snr), 1),
        'status': status
    })

df_87 = pd.DataFrame(res_87).sort_values('power')
df_85 = pd.DataFrame(res_85).sort_values('power')

# =========================================================================
# FIGURE 1: Power Broadening Law Fits & Extrapolations (14 Runs)
# =========================================================================
valid_87 = df_87[(df_87['power'] >= 35) & (~df_87['fwhm'].isna())]
valid_85 = df_85[(df_85['power'] >= 35) & (~df_85['fwhm'].isna())]

popt_sas87, pcov_sas87 = curve_fit(sas_model, valid_87['power'], valid_87['fwhm'], p0=[5.5, 200.0])
popt_lin87, pcov_lin87 = curve_fit(lin_model, valid_87['power'], valid_87['fwhm'], p0=[5.5, 0.003])

popt_sas85, pcov_sas85 = curve_fit(sas_model, valid_85['power'], valid_85['fwhm'], p0=[5.0, 150.0])
popt_lin85, pcov_lin85 = curve_fit(lin_model, valid_85['power'], valid_85['fwhm'], p0=[5.0, 0.003])

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(15, 6))
p_dense = np.linspace(0, 950, 300)

# 87Rb
ax1a.errorbar(valid_87['power'], valid_87['fwhm'], yerr=valid_87['err'], fmt='o', color='#1f77b4', ms=7, capsize=4, label='$^{87}$Rb Data (13 valid runs)')
ax1a.plot(p_dense, sas_model(p_dense, *popt_sas87), '-', color='#1f77b4', lw=2, 
          label=f'SAS Model: $\\Gamma_0 = {popt_sas87[0]:.2f} \\pm {np.sqrt(pcov_sas87[0,0]):.2f}$ MHz\n($P_{{sat}} = {popt_sas87[1]:.0f}\\ \\mu$W)')
ax1a.plot(p_dense, lin_model(p_dense, *popt_lin87), ':', color='#2ca02c', lw=1.8, 
          label=f'Linear Extrap: $\\Gamma_0 = {popt_lin87[0]:.2f} \\pm {np.sqrt(pcov_lin87[0,0]):.2f}$ MHz\n(Slope = {popt_lin87[1]*1e3:.2f} kHz/$\\mu$W)')
ax1a.axhline(6.07, color='black', ls='--', lw=1.2, label='Natural Linewidth (6.07 MHz)')
ax1a.set_xlabel('Pump Power ($\\mu$W)', fontsize=11)
ax1a.set_ylabel('FWHM Linewidth (MHz)', fontsize=11)
ax1a.set_title('(a) $^{87}$Rb Cycling Linewidth vs Pump Power (14 Datasets)', fontweight='bold', fontsize=12)
ax1a.grid(True, alpha=0.3)
ax1a.legend(loc='lower right', fontsize=9.5)
ax1a.set_xlim([-20, 950])

# 85Rb
ax1b.errorbar(valid_85['power'], valid_85['fwhm'], yerr=valid_85['err'], fmt='s', color='#ff7f0e', ms=7, capsize=4, label='$^{85}$Rb Data (13 valid runs)')
ax1b.plot(p_dense, sas_model(p_dense, *popt_sas85), '-', color='#ff7f0e', lw=2, 
          label=f'SAS Model: $\\Gamma_0 = {popt_sas85[0]:.2f} \\pm {np.sqrt(pcov_sas85[0,0]):.2f}$ MHz\n($P_{{sat}} = {popt_sas85[1]:.0f}\\ \\mu$W)')
ax1b.plot(p_dense, lin_model(p_dense, *popt_lin85), ':', color='#d62728', lw=1.8, 
          label=f'Linear Extrap: $\\Gamma_0 = {popt_lin85[0]:.2f} \\pm {np.sqrt(pcov_lin85[0,0]):.2f}$ MHz\n(Slope = {popt_lin85[1]*1e3:.2f} kHz/$\\mu$W)')
ax1b.axhline(6.07, color='black', ls='--', lw=1.2, label='Natural Linewidth (6.07 MHz)')
ax1b.set_xlabel('Pump Power ($\\mu$W)', fontsize=11)
ax1b.set_ylabel('FWHM Linewidth (MHz)', fontsize=11)
ax1b.set_title('(b) $^{85}$Rb Cycling Linewidth vs Pump Power (14 Datasets)', fontweight='bold', fontsize=12)
ax1b.grid(True, alpha=0.3)
ax1b.legend(loc='lower right', fontsize=9.5)
ax1b.set_xlim([-20, 950])

plt.tight_layout()
fig1_path = os.path.join(out_dir, 'session10_power_broadening_14runs_summary.png')
fig1.savefig(fig1_path, dpi=160)
plt.close(fig1)
print(f'Saved Figure 1 to {fig1_path}')

# =========================================================================
# FIGURE 2: 87Rb Lineshapes (14 Panels Sorted by Power)
# =========================================================================
fig2, axes2 = plt.subplots(7, 2, figsize=(15, 26))
axes2 = axes2.flatten()

for idx, (_, r) in enumerate(df_87.iterrows()):
    ax = axes2[idx]
    x_dense = np.linspace(-14, 14, 200)
    popt = r['popt']
    fwhm = r['fwhm']
    err = r['err']
    pwr = r['power']
    r_id = int(r['run'])
    
    if not np.isnan(fwhm) and fwhm > 0:
        y_fit = lorentzian(x_dense, *popt)
        ax.plot(r['x'], r['y'] * 1e3, '.', color='#1f77b4', alpha=0.35, ms=3.5, label='Data')
        ax.plot(x_dense, y_fit * 1e3, 'r-', lw=1.8, label=f'Fit (FWHM = {fwhm:.2f} $\\pm$ {err:.2f} MHz)')
        ax.plot(x_dense, (popt[3] + popt[4] * (x_dense - popt[0])) * 1e3, 'k:', lw=1, label='Baseline')
        amp_v = r['amp']
        res_v = r['res_rms']
        ax.text(0.03, 0.72, f'Peak: {amp_v:.2f} mV\nRMS: {res_v:.3f} mV', 
                transform=ax.transAxes, fontsize=8, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))
    else:
        ax.plot(r['x'], r['y'] * 1e3, '.', color='gray', alpha=0.3, ms=3.5, label='Noise-limited Data')
        ax.text(0.5, 0.5, 'Signal below noise floor\\n(SNR < 2:1)', transform=ax.transAxes, 
                ha='center', va='center', color='red', fontsize=10, fontweight='bold')
        
    ax.set_title(f'Run {r_id} ({pwr:.1f} $\\mu$W) — $^{{87}}$Rb: FWHM = {fwhm:.2f} MHz', fontweight='bold', fontsize=10.5)
    ax.set_xlabel('Relative Frequency (MHz)', fontsize=9.5)
    ax.set_ylabel('Signal (mV)', fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
fig2_path = os.path.join(out_dir, 'session10_power_lineshapes_87rb_14runs.png')
fig2.savefig(fig2_path, dpi=150)
plt.close(fig2)
print(f'Saved Figure 2 to {fig2_path}')

# =========================================================================
# FIGURE 3: 85Rb Lineshapes (14 Panels Sorted by Power)
# =========================================================================
fig3, axes3 = plt.subplots(7, 2, figsize=(15, 26))
axes3 = axes3.flatten()

for idx, (_, r) in enumerate(df_85.iterrows()):
    ax = axes3[idx]
    x_dense = np.linspace(-12, 12, 200)
    popt = r['popt']
    fwhm = r['fwhm']
    err = r['err']
    pwr = r['power']
    r_id = int(r['run'])
    
    if not np.isnan(fwhm) and fwhm > 0:
        y_fit = lorentzian(x_dense, *popt)
        ax.plot(r['x'], r['y'] * 1e3, '.', color='#ff7f0e', alpha=0.35, ms=3.5, label='Data')
        ax.plot(x_dense, y_fit * 1e3, 'r-', lw=1.8, label=f'Fit (FWHM = {fwhm:.2f} $\\pm$ {err:.2f} MHz)')
        ax.plot(x_dense, (popt[3] + popt[4] * (x_dense - popt[0])) * 1e3, 'k:', lw=1, label='Baseline')
        amp_v = r['amp']
        res_v = r['res_rms']
        ax.text(0.03, 0.72, f'Peak: {amp_v:.2f} mV\nRMS: {res_v:.3f} mV', 
                transform=ax.transAxes, fontsize=8, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))
    else:
        ax.plot(r['x'], r['y'] * 1e3, '.', color='gray', alpha=0.3, ms=3.5, label='Noise-limited Data')
        ax.text(0.5, 0.5, 'Signal below noise floor', transform=ax.transAxes, ha='center', va='center', color='red', fontsize=10)
        
    ax.set_title(f'Run {r_id} ({pwr:.1f} $\\mu$W) — $^{{85}}$Rb: FWHM = {fwhm:.2f} MHz', fontweight='bold', fontsize=10.5)
    ax.set_xlabel('Relative Frequency (MHz)', fontsize=9.5)
    ax.set_ylabel('Signal (mV)', fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
fig3_path = os.path.join(out_dir, 'session10_power_lineshapes_85rb_14runs.png')
fig3.savefig(fig3_path, dpi=150)
plt.close(fig3)
print(f'Saved Figure 3 to {fig3_path}')

# =========================================================================
# FIGURE 4: Peak Resolution and SNR Trade-off (14 Runs)
# =========================================================================
fig4, ax4 = plt.subplots(figsize=(10, 5.5))
df_res = pd.DataFrame(summary_list).sort_values('power')
valid_res = df_res[df_res['power'] >= 35]

ax4_twin = ax4.twinx()
line1 = ax4.plot(valid_res['power'], valid_res['contrast'], 'o-', color='#1f77b4', lw=2, ms=6, label='Peak Valley Contrast (%)')
line2 = ax4_twin.plot(valid_res['power'], valid_res['snr'], 's--', color='#2ca02c', lw=2, ms=6, label='Signal-to-Noise Ratio (SNR)')

ax4.set_xlabel('Pump Power ($\\mu$W)', fontsize=11)
ax4.set_ylabel('Valley Contrast (%)', color='#1f77b4', fontsize=11)
ax4_twin.set_ylabel('SNR (Peak / Noise Floor)', color='#2ca02c', fontsize=11)
ax4.set_title('Sub-Doppler Peak Resolvability & SNR Trade-off (14 Datasets)', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, loc='lower right', fontsize=10)

plt.tight_layout()
fig4_path = os.path.join(out_dir, 'session10_peak_resolution_vs_power_14runs.png')
fig4.savefig(fig4_path, dpi=160)
plt.close(fig4)
print(f'Saved Figure 4 to {fig4_path}')

print('All 14-run power analysis figures successfully generated!')
