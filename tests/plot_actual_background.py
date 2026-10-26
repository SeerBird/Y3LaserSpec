import sys
import os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

fsr = 374.74
out_dir = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260'

def load_and_calibrate(r_id):
    df_p = pd.read_csv(f'data/Session 10/Power variation/P{r_id}.CSV')
    df_u = pd.read_csv(f'data/Session 10/Power variation/U{r_id}.CSV')
    f_path = f'data/Session 10/Power variation/F{r_id}.CSV'
    if not os.path.exists(f_path): f_path = 'data/Session 10/Power variation/F14.CSV'
    df_f = pd.read_csv(f_path)
    
    t = df_p.iloc[:, 0].values
    p = df_p.iloc[:, 1].values
    u = df_u.iloc[:, 1].values
    diff = p - u
    fp = df_f.iloc[:, 1].values
    
    peaks_fp, _ = signal.find_peaks(fp, height=-0.045, prominence=0.005, distance=1000)
    poly = np.poly1d(np.polyfit(t[peaks_fp], np.arange(len(peaks_fp)), 4))
    nu = poly(t) * fsr
    return t, nu, p, u, diff

# Let's inspect Run 1 (max power 859 uW) and Run 5 (medium power 300 uW)
t1, nu1, p1, u1, diff1 = load_and_calibrate(1)
t5, nu5, p5, u5, diff5 = load_and_calibrate(5)

fig = plt.figure(figsize=(15, 12), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.1])

# Panel 1 (Top, span both cols): Full Scan Differential Signal
ax_full = fig.add_subplot(gs[0, :])
ax_full.plot(nu1, diff1 * 1e3, color='#1f77b4', lw=1.0, label='Run 1 (859.0 $\\mu$W) Diff $(P - U)$')
ax_full.plot(nu5, diff5 * 1e3, color='#2ca02c', lw=0.9, alpha=0.85, label='Run 5 (299.5 $\\mu$W) Diff $(P - U)$')
ax_full.axhline(0, color='black', ls='--', lw=1.0, alpha=0.6)

# Annotate the 4 manifolds
manifolds = [
    ('$^{87}$Rb $F=2$', 750, 48),
    ('$^{85}$Rb $F=3$', 1980, 92),
    ('$^{85}$Rb $F=2$', 4930, 65),
    ('$^{87}$Rb $F=1$', 7400, 22)
]
for name, pos, h in manifolds:
    ax_full.annotate(name, xy=(pos, h), xytext=(pos, h + 12),
                     ha='center', fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='#d62728'))

# Shade empty background regions
ax_full.axvspan(2400, 4600, color='#ffffcc', alpha=0.4, label='Inter-Manifold Baseline Gap (2.2 GHz wide)')
ax_full.axvspan(5400, 7000, color='#ffffcc', alpha=0.4)
ax_full.set_title('(a) Full Frequency Scan of Differential Transmission $(P - U)$ across all 4 Manifolds', fontsize=12, fontweight='bold')
ax_full.set_xlabel('Relative Frequency $\\nu$ (MHz)', fontsize=11)
ax_full.set_ylabel('Differential Signal (mV)', fontsize=11)
ax_full.set_xlim(-100, 8100)
ax_full.grid(True, alpha=0.3)
ax_full.legend(loc='upper right', fontsize=9.5)

# Panel 2 (Middle Left): Zoom on the "Empty" Baseline Gap (2400 to 4600 MHz)
ax_gap = fig.add_subplot(gs[1, 0])
gap_mask1 = (nu1 >= 2400) & (nu1 <= 4600)
ax_gap.plot(nu1[gap_mask1], diff1[gap_mask1] * 1e3, color='#1f77b4', lw=0.9, alpha=0.7, label='Run 1 (859 $\\mu$W)')
gap_mask5 = (nu5 >= 2400) & (nu5 <= 4600)
ax_gap.plot(nu5[gap_mask5], diff5[gap_mask5] * 1e3, color='#2ca02c', lw=0.9, alpha=0.7, label='Run 5 (300 $\\mu$W)')
ax_gap.axhline(0, color='red', ls='--', lw=1.2, label='Zero Level')
ax_gap.axhline(0.16, color='gray', ls=':', lw=0.9, label='Noise Floor ($\\pm 0.16$ mV)')
ax_gap.axhline(-0.16, color='gray', ls=':', lw=0.9)
ax_gap.set_title('(b) Baseline Zoom in "Empty" Spectral Region (2400–4600 MHz)', fontsize=11, fontweight='bold')
ax_gap.set_xlabel('Relative Frequency $\\nu$ (MHz)', fontsize=10)
ax_gap.set_ylabel('Signal (mV)', fontsize=10)
ax_gap.set_ylim(-3.5, 3.5)
ax_gap.grid(True, alpha=0.3)
ax_gap.legend(loc='lower right', fontsize=8.5)

# Panel 3 (Middle Right): Raw Channels P and U (showing Doppler Profile)
ax_raw = fig.add_subplot(gs[1, 1])
sub_mask1 = (nu1 >= 300) & (nu1 <= 2500)
ax_raw.plot(nu1[sub_mask1], p1[sub_mask1], color='#e377c2', lw=1.1, label='Pumped $P(\\nu)$')
ax_raw.plot(nu1[sub_mask1], u1[sub_mask1], color='#7f7f7f', lw=1.1, ls='--', label='Unpumped $U(\\nu)$ (Doppler Dips)')
ax_raw.set_title('(c) Raw Detector Signals: Pumped vs Unpumped Doppler Profiles', fontsize=11, fontweight='bold')
ax_raw.set_xlabel('Relative Frequency $\\nu$ (MHz)', fontsize=10)
ax_raw.set_ylabel('Photodetector Voltage (V)', fontsize=10)
ax_raw.grid(True, alpha=0.3)
ax_raw.legend(loc='lower left', fontsize=9)

# Panel 4 (Bottom Left): Zoom on 87Rb F=2 Manifold Baseline Pedestal
ax_m87 = fig.add_subplot(gs[2, 0])
m87_mask = (nu1 >= 400) & (nu1 <= 1100)
ax_m87.plot(nu1[m87_mask], diff1[m87_mask] * 1e3, color='#1f77b4', lw=1.2, label='Run 1 (859 $\\mu$W)')
ax_m87.plot(nu5[m87_mask], diff5[m87_mask] * 1e3, color='#2ca02c', lw=1.2, label='Run 5 (300 $\\mu$W)')
ax_m87.axhline(0, color='red', ls='--', lw=1.0)
ax_m87.set_title('(d) $^{87}$Rb $F=2$ Manifold: Sub-Doppler Peaks & Wings', fontsize=11, fontweight='bold')
ax_m87.set_xlabel('Relative Frequency $\\nu$ (MHz)', fontsize=10)
ax_m87.set_ylabel('Signal (mV)', fontsize=10)
ax_m87.grid(True, alpha=0.3)
ax_m87.legend(loc='upper right', fontsize=9)

# Panel 5 (Bottom Right): Zoom on 85Rb F=3 Manifold Baseline Pedestal
ax_m85 = fig.add_subplot(gs[2, 1])
m85_mask = (nu1 >= 1700) & (nu1 <= 2200)
ax_m85.plot(nu1[m85_mask], diff1[m85_mask] * 1e3, color='#1f77b4', lw=1.2, label='Run 1 (859 $\\mu$W)')
ax_m85.plot(nu5[m85_mask], diff5[m85_mask] * 1e3, color='#2ca02c', lw=1.2, label='Run 5 (300 $\\mu$W)')
ax_m85.axhline(0, color='red', ls='--', lw=1.0)
ax_m85.set_title('(e) $^{85}$Rb $F=3$ Manifold: Sub-Doppler Peaks & Wings', fontsize=11, fontweight='bold')
ax_m85.set_xlabel('Relative Frequency $\\nu$ (MHz)', fontsize=10)
ax_m85.set_ylabel('Signal (mV)', fontsize=10)
ax_m85.grid(True, alpha=0.3)
ax_m85.legend(loc='upper right', fontsize=9)

fig_path = os.path.join(out_dir, 'session10_actual_background_inspection.png')
plt.savefig(fig_path, dpi=160)
plt.close()
print(f'Saved background inspection plot to {fig_path}')
