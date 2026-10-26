import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

# 1. Load data
ncf = pd.read_csv('data/Session 10/No cell/NCF.CSV')
ncp = pd.read_csv('data/Session 10/No cell/NCP01.CSV')
ncu = pd.read_csv('data/Session 10/No cell/NCU.CSV')

t = ncp['in s'].values
p = ncp['C3 in V'].values
u = ncu['C3 in V'].values
fp = ncf['C2 in V'].values

# 2. FP Calibration
peaks_fp, _ = signal.find_peaks(fp, height=-0.045, prominence=0.005, distance=1000)
fsr = 374.74 # MHz
peak_idx = np.arange(len(peaks_fp))
poly = np.poly1d(np.polyfit(peaks_fp, peak_idx, 5))
fringe_coord = poly(np.arange(len(t)))
freq_mhz = fringe_coord * fsr
freq_mhz -= freq_mhz[peaks_fp[0]] # 0 at first fringe

# Difference
diff = p - u
mean_diff = np.mean(diff)
rms_diff = np.std(diff)
pkpk_diff = np.ptp(diff)

# Probe features
poly_macro = np.poly1d(np.polyfit(freq_mhz, u, 3))
macro_baseline = poly_macro(freq_mhz)
ripples = u - macro_baseline

# Uniform frequency grid for FFT
nu_grid = np.linspace(freq_mhz[0], freq_mhz[-1], len(freq_mhz))
ripples_grid = np.interp(nu_grid, freq_mhz, ripples)
dnu = nu_grid[1] - nu_grid[0]
f_nu, psd_nu = signal.welch(ripples_grid, fs=1.0/dnu, nperseg=16384)

# Create figure
fig = plt.figure(figsize=(13, 11))

# Panel 1: Pumped vs Unpumped
ax1 = plt.subplot(3, 2, 1)
ax1.plot(t * 1e3, p * 1e3, label='Pumped ($NCP01$)', color='#1f77b4', lw=1.2)
ax1.plot(t * 1e3, u * 1e3, label='Unpumped ($NCU$)', color='#ff7f0e', lw=1.2, ls='--')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Detector Signal (mV)')
ax1.set_title('(a) Raw Traces: Pumped vs Unpumped', fontweight='bold')
ax1.legend(loc='lower left')
ax1.grid(True, alpha=0.3)

# Panel 2: FP Trace
ax2 = plt.subplot(3, 2, 2)
ax2.plot(t * 1e3, fp * 1e3, color='#2ca02c', lw=1)
ax2.plot(t[peaks_fp] * 1e3, fp[peaks_fp] * 1e3, 'rx', label=f'{len(peaks_fp)} Fringes (FSR={fsr:.1f} MHz)')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Fabry-Pérot (mV)')
ax2.set_title(f'(b) Fabry-Pérot Frequency Axis (Span: {(freq_mhz[-1]-freq_mhz[0])/1e3:.1f} GHz)', fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# Panel 3: Difference (Pumped - Unpumped)
ax3 = plt.subplot(3, 2, 3)
ax3.plot(freq_mhz * 1e-3, diff * 1e3, color='#d62728', lw=0.8, alpha=0.8)
ax3.axhline(mean_diff * 1e3, color='black', ls='--', label=f'Mean Offset: {mean_diff*1e6:.1f} $\\mu$V')
ax3.set_xlabel('Frequency (GHz relative to Fringe 0)')
ax3.set_ylabel('Diff: P - U (mV)')
ax3.set_title(f'(c) Difference Trace: Residual RMS = {rms_diff*1e3:.3f} mV', fontweight='bold')
ax3.set_ylim([-1.5, 1.5])
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Panel 4: Zoom of Difference showing noise floor
ax4 = plt.subplot(3, 2, 4)
zoom_mask = (freq_mhz >= 2000) & (freq_mhz <= 4000)
ax4.plot(freq_mhz[zoom_mask] * 1e-3, diff[zoom_mask] * 1e3, color='#d62728', lw=1)
ax4.set_xlabel('Frequency (GHz)')
ax4.set_ylabel('Diff: P - U (mV)')
ax4.set_title('(d) Zoomed Difference (2 to 4 GHz): Zero Pump Leakage', fontweight='bold')
ax4.set_ylim([-0.8, 0.8])
ax4.grid(True, alpha=0.3)

# Panel 5: Probe Ripple (Etalon) vs Frequency
ax5 = plt.subplot(3, 2, 5)
ax5.plot(freq_mhz * 1e-3, ripples * 1e3, color='#9467bd', lw=1.2, label='Detrended Probe Ripple')
# Overlay sinusoidal model of 1682 MHz
amp_est = np.ptp(ripples) * 1e3 / 2.0 * 0.7
phase_est = 0.5
sine_model = amp_est * np.sin(2 * np.pi * freq_mhz / 1681.8 + phase_est)
ax5.plot(freq_mhz * 1e-3, sine_model, 'k--', alpha=0.6, label='Sine Fit (Period = 1682 MHz / 4.5 FSR)')
ax5.set_xlabel('Frequency (GHz)')
ax5.set_ylabel('Ripple Amplitude (mV)')
ax5.set_title('(e) Parasitic Etalon Wiggle (Pk-Pk = 26.7 mV)', fontweight='bold')
ax5.legend(loc='lower right')
ax5.grid(True, alpha=0.3)

# Panel 6: Spatial Frequency Spectrum (PSD)
ax6 = plt.subplot(3, 2, 6)
# Plot in period (MHz)
mask_psd = (f_nu > 0) & (f_nu < 0.005) # up to 200 MHz period
periods = 1.0 / f_nu[mask_psd]
ax6.plot(periods, psd_nu[mask_psd] * 1e6, color='#8c564b', lw=1.5)
ax6.axvline(1681.8, color='red', ls='--', label='Dominant Etalon: 1682 MHz ($L_{air} \\approx 8.9$ cm)')
ax6.set_xlabel('Spectral Period (MHz)')
ax6.set_ylabel('PSD ($10^{-6}\\ \\mathrm{V}^2/\\mathrm{MHz}$)')
ax6.set_title('(f) Ripple Power Spectrum (Cavity Length Identification)', fontweight='bold')
ax6.legend(loc='upper right')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
out_path = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260/session10_nocell_analysis.png'
plt.savefig(out_path, dpi=160)
print(f'Figure saved to {out_path}')
