import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

fsr = 374.74  # MHz

s9_dir = r'data\Session 9'
f_files = [f'F{i}.CSV' for i in range(1, 8)]

fig_dir = r'figures\fp_analysis'
artifact_dir = r'C:\Users\ilyag\.gemini\antigravity\brain\a9066794-3e7c-4948-9556-40d00e7a0260'
os.makedirs(fig_dir, exist_ok=True)

fp_traces = {}

for fname in f_files:
    path = os.path.join(s9_dir, fname)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    t = df.iloc[:, 0].values
    v = df.iloc[:, -1].values
    peaks, _ = find_peaks(v, prominence=0.015, distance=100)
    
    # Fit polynomial to fringe order vs sample index
    peak_idx = np.arange(len(peaks))
    poly_fit = np.poly1d(np.polyfit(peaks, peak_idx, 5))
    
    # Also fit polynomial to fringe order vs time
    poly_fit_t = np.poly1d(np.polyfit(t[peaks], peak_idx, 5))
    
    # Inter-fringe time differences
    dt_fringes = np.diff(t[peaks])
    
    fp_traces[fname] = {
        't': t,
        'v': v,
        'peaks': peaks,
        'peak_order': peak_idx,
        't_peaks': t[peaks],
        'dt_fringes': dt_fringes,
        'poly_fit': poly_fit,
        'poly_fit_t': poly_fit_t,
        'n_fringes': len(peaks)
    }

# Compare inter-fringe spacing dt across runs
print("\n--- Inter-fringe time spacing stats (ms) ---")
for fname, tr in fp_traces.items():
    dt_ms = tr['dt_fringes'] * 1e3
    print(f"{fname:8s}: {tr['n_fringes']} fringes | mean dt = {dt_ms.mean():.4f} ms, std = {dt_ms.std():.4f} ms, range [{dt_ms.min():.4f}, {dt_ms.max():.4f}] ms")

# Let's compare files with similar sweep duration: F1, F3, F4, F6, F7
std_files = ['F1.CSV', 'F3.CSV', 'F4.CSV', 'F6.CSV', 'F7.CSV']

# Plot 1: Inter-fringe spacing (dt) vs fringe order (Piezo nonlinearity)
plt.figure(figsize=(12, 6))
for fname in f_files:
    tr = fp_traces[fname]
    dt_ms = tr['dt_fringes'] * 1e3
    plt.plot(np.arange(len(dt_ms)), dt_ms, 'o-', ms=4, label=f"{fname} ({tr['n_fringes']} fringes)")
plt.xlabel('Fringe Order', fontsize=12)
plt.ylabel('Inter-fringe Time Spacing $\\Delta t$ (ms)', fontsize=12)
plt.title('Fabry-Pérot Inter-fringe Time Spacing across Session 9 Runs', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'interfringe_spacing_all_runs.png'), dpi=200)
plt.savefig(os.path.join(artifact_dir, 'interfringe_spacing_all_runs.png'), dpi=200)
plt.close()

# Plot 2: Fringe peak position drift in time
plt.figure(figsize=(12, 6))
for fname in std_files:
    tr = fp_traces[fname]
    # Plot fringe order vs time
    plt.plot(tr['t_peaks'] * 1e3, tr['peak_order'], 'o-', ms=4, label=fname)
plt.xlabel('Time on Oscilloscope (ms)', fontsize=12)
plt.ylabel('Fringe Order $n$', fontsize=12)
plt.title('Fringe Order vs. Time (Trigger Jitter & Time Drift across Standard Runs)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fringe_order_vs_time.png'), dpi=200)
plt.savefig(os.path.join(artifact_dir, 'fringe_order_vs_time.png'), dpi=200)
plt.close()

# Quantify Frequency Error if F1 calibration is applied to F3, F4, F6, F7
# Suppose we align in time: shift t so that the 10th fringe lines up with F1's 10th fringe
f1 = fp_traces['F1.CSV']
t_f1_10 = f1['t_peaks'][10]

print("\n--- Frequency Axis Discrepancy if F1 calibration is borrowed (MHz) ---")
plt.figure(figsize=(12, 6))

for fname in ['F3.CSV', 'F4.CSV', 'F6.CSV', 'F7.CSV']:
    tr = fp_traces[fname]
    # Time shift to align the 10th fringe to F1
    t_shift = t_f1_10 - tr['t_peaks'][10]
    t_aligned = tr['t_peaks'] + t_shift
    
    # What fringe order does F1's calibration assign to these peaks?
    n_borrowed = f1['poly_fit_t'](t_aligned)
    n_true = tr['peak_order']
    
    # Error in fringes and MHz
    dn_fringe = n_borrowed - n_true
    df_mhz = dn_fringe * fsr
    
    print(f"{fname:8s}: time shift = {t_shift*1e3:6.2f} ms | Max freq error = {np.max(np.abs(df_mhz)):5.2f} MHz, RMS = {np.sqrt(np.mean(df_mhz**2)):5.2f} MHz")
    plt.plot(n_true, df_mhz, 'o-', ms=4, label=f"{fname} (shift {t_shift*1e3:.1f} ms, max err {np.max(np.abs(df_mhz)):.1f} MHz)")

plt.axhline(0, color='k', ls='--', lw=0.8)
plt.xlabel('Fringe Order $n$', fontsize=12)
plt.ylabel('Frequency Calibration Discrepancy (MHz)', fontsize=12)
plt.title('Frequency Axis Error when Borrowing F1 Calibration (Time-Shifted)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'calibration_error_when_borrowing.png'), dpi=200)
plt.savefig(os.path.join(artifact_dir, 'calibration_error_when_borrowing.png'), dpi=200)
plt.close()

# Also compare Magnetic Variation runs GF1 to GF5
gf_files = [f'GF{i}.CSV' for i in range(1, 6)]
gf_traces = {}
for fname in gf_files:
    path = os.path.join(r'data\Session 9\Magnetic variation', fname)
    df = pd.read_csv(path)
    t = df.iloc[:, 0].values
    v = df.iloc[:, -1].values
    peaks, _ = find_peaks(v, prominence=0.015, distance=100)
    gf_traces[fname] = {
        't': t, 'v': v, 'peaks': peaks, 't_peaks': t[peaks],
        'poly_fit_t': np.poly1d(np.polyfit(t[peaks], np.arange(len(peaks)), 5)),
        'peak_order': np.arange(len(peaks))
    }

print("\n--- Magnetic Variation: Borrowing GF1 calibration for GF2-5 ---")
gf1 = gf_traces['GF1.CSV']
t_gf1_10 = gf1['t_peaks'][10]

for fname in ['GF2.CSV', 'GF3.CSV', 'GF4.CSV', 'GF5.CSV']:
    tr = gf_traces[fname]
    t_shift = t_gf1_10 - tr['t_peaks'][10]
    t_aligned = tr['t_peaks'] + t_shift
    n_borrowed = gf1['poly_fit_t'](t_aligned)
    df_mhz = (n_borrowed - tr['peak_order']) * fsr
    print(f"{fname:8s}: time shift = {t_shift*1e3:6.2f} ms | Max freq error = {np.max(np.abs(df_mhz)):5.2f} MHz, RMS = {np.sqrt(np.mean(df_mhz**2)):5.2f} MHz")

print("\nAnalysis script finished successfully!")
