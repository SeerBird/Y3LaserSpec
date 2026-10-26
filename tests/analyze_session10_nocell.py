import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import interp1d

# 1. Load data
ncf = pd.read_csv('data/Session 10/No cell/NCF.CSV')
ncp = pd.read_csv('data/Session 10/No cell/NCP01.CSV')
ncu = pd.read_csv('data/Session 10/No cell/NCU.CSV')

t = ncp['in s'].values
p = ncp['C3 in V'].values
u = ncu['C3 in V'].values
fp = ncf['C2 in V'].values

# 2. Fabry-Perot fringe detection
peaks, _ = signal.find_peaks(fp, height=-0.045, prominence=0.005, distance=1000)
fsr = 374.74  # MHz nominal FSR
peak_idx = np.arange(len(peaks))
poly = np.poly1d(np.polyfit(peaks, peak_idx, 5))
fringe_coord = poly(np.arange(len(t)))
freq_mhz = fringe_coord * fsr
freq_mhz -= freq_mhz[peaks[0]]

# Align u to p with small sub-sample shift to see if residual diff is jitter
# Fast cross-correlation / minimization
idx_eval = np.arange(2000, len(t) - 2000)
from scipy.optimize import minimize_scalar
u_interp = interp1d(np.arange(len(t)), u, kind='cubic', bounds_error=False, fill_value='extrapolate')
res = minimize_scalar(lambda s: np.mean((p[idx_eval] - u_interp(idx_eval + s))**2), bounds=(-50, 50), method='bounded')
u_aligned = u_interp(np.arange(len(t)) + res.x)

raw_diff = p - u
aligned_diff = p - u_aligned

print(f"Optimal alignment shift: {res.x:.3f} samples ({(res.x*(t[1]-t[0])*1e6):.2f} us)")
print(f"Raw diff std: {raw_diff.std()*1e3:.4f} mV, pk-pk: {(raw_diff.max()-raw_diff.min())*1e3:.4f} mV")
print(f"Aligned diff std: {aligned_diff.std()*1e3:.4f} mV, pk-pk: {(aligned_diff.max()-aligned_diff.min())*1e3:.4f} mV")

# 3. Create comprehensive 4-panel diagnostic figure
fig = plt.figure(figsize=(14, 12))

# Subplot 1: Raw Traces
ax1 = plt.subplot(4, 1, 1)
ax1.plot(t * 1e3, p * 1e3, label='Pumped (NCP01)', color='#1f77b4', lw=1.2)
ax1.plot(t * 1e3, u * 1e3, label='Unpumped (NCU)', color='#ff7f0e', lw=1.2, ls='--')
ax1.set_ylabel('Detector (mV)')
ax1.set_title('Session 10 "No Cell" Traces: Pumped vs Unpumped', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Subplot 2: Fabry-Perot Interferometer
ax2 = plt.subplot(4, 1, 2, sharex=ax1)
ax2.plot(t * 1e3, fp * 1e3, label='Fabry-Perot (NCF)', color='#2ca02c', lw=1)
ax2.plot(t[peaks] * 1e3, fp[peaks] * 1e3, 'rx', label=f'Detected Fringes ({len(peaks)} total, FSR = {fsr:.2f} MHz)')
ax2.set_ylabel('FP (mV)')
ax2.set_title(f'Fabry-Pérot Frequency Calibration ({len(peaks)} fringes across scan)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# Subplot 3: Difference (Pumped - Unpumped) vs Time
ax3 = plt.subplot(4, 1, 3, sharex=ax1)
ax3.plot(t * 1e3, raw_diff * 1e3, label='Raw Diff (P - U)', color='#d62728', lw=1, alpha=0.7)
ax3.plot(t * 1e3, aligned_diff * 1e3, label=f'Aligned Diff (Shift = {res.x:.2f} samples)', color='#9467bd', lw=1.2)
ax3.axhline(0, color='black', lw=0.8, ls=':')
ax3.set_ylabel('Diff (mV)')
ax3.set_title(f'Difference (Pumped - Unpumped): Residual pk-pk = {(aligned_diff.max()-aligned_diff.min())*1e3:.2f} mV, RMS = {aligned_diff.std()*1e3:.3f} mV', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

# Subplot 4: Difference vs Calibrated Frequency (MHz)
ax4 = plt.subplot(4, 1, 4)
ax4.plot(freq_mhz * 1e-3, aligned_diff * 1e3, color='#9467bd', lw=1.2)
ax4.axhline(0, color='black', lw=0.8, ls=':')
ax4.set_xlabel('Calibrated Laser Frequency (GHz relative to first fringe)')
ax4.set_ylabel('Diff (mV)')
ax4.set_title('Aligned Difference vs Frequency (No Sub-Doppler Dips Visible Without Cell)', fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
out_fig = 'tests/session10_nocell_analysis.png'
plt.savefig(out_fig, dpi=150)
print(f'Saved figure to {out_fig}')
