import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

csv_path = r'd:\FINRUN\NOCELL.CSV'
fig_dir = r'figures\nocell'
artifact_dir = r'C:\Users\ilyag\.gemini\antigravity\brain\a9066794-3e7c-4948-9556-40d00e7a0260'
os.makedirs(fig_dir, exist_ok=True)

df = pd.read_csv(csv_path)
t = df.iloc[:, 0].values
v = df.iloc[:, -1].values

print(f"Loaded {csv_path}: {len(t)} points")
print(f"Time range: [{t.min():.5f}, {t.max():.5f}] s (duration = {(t.max() - t.min())*1e3:.2f} ms)")
print(f"Voltage range: [{v.min():.5f}, {v.max():.5f}] V, mean = {v.mean():.5f} V, std = {v.std():.5f} V")

# Check sampling interval
dt = np.mean(np.diff(t))
fs = 1.0 / dt
print(f"dt = {dt*1e6:.3f} us, sampling rate = {fs/1e3:.2f} kHz")

# FFT of the voltage signal
# Detrend by subtracting a polynomial background (e.g. laser power ramp)
poly_bg = np.poly1d(np.polyfit(t, v, 2))
v_detrend = v - poly_bg(t)

fft_vals = np.fft.rfft(v_detrend)
fft_freqs = np.fft.rfftfreq(len(t), d=dt)
fft_power = np.abs(fft_vals) ** 2

# Find dominant peaks in FFT
fft_pks, _ = find_peaks(fft_power, prominence=np.max(fft_power)*0.05, distance=10)
top_pks = sorted(fft_pks, key=lambda idx: fft_power[idx], reverse=True)[:5]

print("\n--- Dominant Ripple Frequencies in Time Domain ---")
for p_idx in top_pks:
    freq_hz = fft_freqs[p_idx]
    period_ms = (1.0 / freq_hz) * 1e3
    print(f"Frequency = {freq_hz:8.2f} Hz  |  Period = {period_ms:8.3f} ms  |  Power = {fft_power[p_idx]:.2e}")

# Check peaks in the time domain to measure ripple period
v_smooth = pd.Series(v).rolling(150, center=True).mean().values
v_smooth = np.nan_to_num(v_smooth, nan=v[0])
v_res = v - v_smooth

# Plot: 3-panel inspection
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

# 1. Full trace
ax1.plot(t * 1e3, v, color='navy', lw=0.8, alpha=0.85, label='Raw Detector Signal (No Cell)')
ax1.plot(t * 1e3, poly_bg(t), 'r--', lw=1.5, label='2nd-order Polynomial Trend (Laser Sweep Ramp)')
ax1.set_ylabel('Voltage (V)', fontsize=11)
ax1.set_title(r'NOCELL.CSV Detector Trace: $V(t)$ with Laser Scanning', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Zoomed in window showing the periodic waviness
# Pick a 10 ms window in the middle
t_mid = 0.5 * (t.min() + t.max())
zoom_mask = (t >= t_mid - 0.005) & (t <= t_mid + 0.005)

ax2.plot(t[zoom_mask] * 1e3, v[zoom_mask], 'o-', ms=1.5, color='royalblue', lw=1.0, label='Zoomed Data')
ax2.plot(t[zoom_mask] * 1e3, poly_bg(t[zoom_mask]), 'r--', lw=1.2, label='Ramp Trend')
ax2.set_ylabel('Voltage (V)', fontsize=11)
ax2.set_title('Zoomed-in View (10 ms window): Periodic Waviness Structure', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. FFT Power Spectrum
ax3.plot(fft_freqs, fft_power, color='crimson', lw=1.0)
for p_idx in top_pks[:3]:
    f_val = fft_freqs[p_idx]
    ax3.annotate(f"{f_val:.1f} Hz\n({(1/f_val)*1e3:.2f} ms)",
                 (f_val, fft_power[p_idx]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
ax3.set_xlim(0, max(fft_freqs[top_pks[0]] * 4, 2000))
ax3.set_xlabel('Frequency (Hz)', fontsize=11)
ax3.set_ylabel('FFT Power (arb. units)', fontsize=11)
ax3.set_title('Power Spectral Density: Identifying Ripple Frequencies', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'nocell_trace_analysis.png'), dpi=200)
plt.savefig(os.path.join(artifact_dir, 'nocell_trace_analysis.png'), dpi=200)
plt.close()

print("Plot saved to nocell_trace_analysis.png successfully!")
