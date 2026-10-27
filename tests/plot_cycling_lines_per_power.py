"""
plot_cycling_lines_per_power.py
-------------------------------
Generates a dedicated figure for each of the 22 power variation runs.
Each figure contains exactly TWO subplots:
  1. 87Rb (F=2 -> F'=3) Cycling Peak with isolated Lorentzian fit & local baseline
  2. 85Rb (F=3 -> F'=4) Cycling Peak with isolated Lorentzian fit & local baseline

Saves figures sorted by power to 'figures/cycling_power_runs/'.
"""

import sys
import os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, curve_fit

# Configuration & paths
fsr = 374.74 # MHz/fringe
out_dir_local = 'figures/cycling_power_runs'
out_dir_brain = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260/cycling_runs'
os.makedirs(out_dir_local, exist_ok=True)
os.makedirs(out_dir_brain, exist_ok=True)

# Load powers
df_pwr = pd.read_excel('data/Session 10/Power variation/Powers.xlsx')
powers = df_pwr['Power_uW'].values
runs = np.arange(1, len(powers) + 1)

# Sort runs by increasing power for convenience
sort_idx = np.argsort(powers)
sorted_runs = runs[sort_idx]
sorted_powers = powers[sort_idx]

def load_and_calibrate_shifted(r_id):
    """Loads Pumped, Unpumped, and FP traces, applies shift alignment, and calibrates frequency."""
    df_p = pd.read_csv(f'data/Session 10/Power variation/P{r_id}.CSV')
    df_u = pd.read_csv(f'data/Session 10/Power variation/U{r_id}.CSV')
    f_path = f'data/Session 10/Power variation/F{r_id}.CSV'
    if not os.path.exists(f_path):
        f_path = 'data/Session 10/Power variation/F14.CSV'
    df_f = pd.read_csv(f_path)
    
    t = df_p.iloc[:, 0].values
    p = df_p.iloc[:, 1].values
    u = df_u.iloc[:, 1].values
    index = np.arange(len(p))
    
    # Background shift optimization
    u_interp = interp1d(index, u, kind='cubic', bounds_error=False, fill_value='extrapolate')
    pad = 2000
    idx_eval = index[pad:-pad]
    p_eval = p[pad:-pad]
    bg_mask = np.abs(p_eval - u[pad:-pad]) < 0.02
    res = minimize_scalar(lambda s: np.mean((p_eval[bg_mask] - u_interp(idx_eval[bg_mask] + s)) ** 2),
                          bounds=(-300, 300), method='bounded')
    u_shifted = u_interp(index + res.x)
    diff = (p - u_shifted) * 1e3 # in mV
    
    # FP peak detection & 4th-order polynomial frequency calibration
    peaks_fp, _ = signal.find_peaks(df_f.iloc[:, 1].values, height=-0.045, prominence=0.005, distance=1000)
    poly = np.poly1d(np.polyfit(t[peaks_fp], np.arange(len(peaks_fp)), 4))
    nu = poly(t) * fsr
    return t, nu, diff

def lorentz_lin(x, x0, gamma, A, c0, c1):
    """Lorentzian peak with a linear background."""
    return c0 + c1 * x + A / (1.0 + 4.0 * ((x - x0) / gamma) ** 2)

def plot_power_run(r_id, pwr, show=True):
    """Fits and plots the two cycling peaks for a single power run."""
    t, nu, diff = load_and_calibrate_shifted(r_id)
    
    # 1. 87Rb F=2 Cycling Peak (F'=3) located around t in [-0.0036, -0.0031]
    m0 = (t >= -0.0036) & (t <= -0.0031)
    t_ref0 = t[m0][np.argmax(diff[m0])]
    nu_ref0 = nu[np.argmin(np.abs(t - t_ref0))]
    
    # 2. 85Rb F=3 Cycling Peak (F'=4) anchored from prominent CO 3-4 peak (+60.32 MHz)
    m_co = (t >= 0.00030) & (t <= 0.00060)
    t_co = t[m_co][np.argmax(diff[m_co])]
    nu_co = nu[np.argmin(np.abs(t - t_co))]
    nu_ref1 = nu_co + 60.32
    
    # --- 87Rb F'=3 ---
    # Isolated Fitting Window: [-20, +20] MHz
    span87 = 20.0
    mask87_fit = (nu >= nu_ref0 - span87) & (nu <= nu_ref0 + span87)
    x87_fit = nu[mask87_fit] - nu_ref0
    y87_fit = diff[mask87_fit]
    p0_87 = [0.0, 7.0, np.max(y87_fit) - np.min(y87_fit), np.min(y87_fit), 0.0]
    popt87, pcov87 = curve_fit(lorentz_lin, x87_fit, y87_fit, p0=p0_87,
                               bounds=([-5.0, 1.0, 0.0, -15.0, -1.0], [5.0, 40.0, 100.0, 15.0, 1.0]))
    perr87 = np.sqrt(np.diag(pcov87))
    res87 = y87_fit - lorentz_lin(x87_fit, *popt87)
    rms87 = np.sqrt(np.mean(res87 ** 2))
    
    # Display Window for 87Rb F=2: Full Manifold [-460, +40] MHz
    mask87_disp = (nu >= nu_ref0 - 460.0) & (nu <= nu_ref0 + 40.0)
    x87_disp = nu[mask87_disp] - nu_ref0
    y87_disp = diff[mask87_disp]
    
    # --- 85Rb F'=4 ---
    # Isolated Fitting Window: [-18, +18] MHz
    span85 = 18.0
    mask85_fit = (nu >= nu_ref1 - span85) & (nu <= nu_ref1 + span85)
    x85_fit = nu[mask85_fit] - nu_ref1
    y85_fit = diff[mask85_fit]
    p0_85 = [0.0, 7.0, np.max(y85_fit) - np.min(y85_fit), np.min(y85_fit), 0.0]
    popt85, pcov85 = curve_fit(lorentz_lin, x85_fit, y85_fit, p0=p0_85,
                               bounds=([-5.0, 1.0, 0.0, -15.0, -1.0], [5.0, 40.0, 100.0, 15.0, 1.0]))
    perr85 = np.sqrt(np.diag(pcov85))
    res85 = y85_fit - lorentz_lin(x85_fit, *popt85)
    rms85 = np.sqrt(np.mean(res85 ** 2))
    
    # Display Window for 85Rb F=3: Full Manifold [-220, +40] MHz
    mask85_disp = (nu >= nu_ref1 - 220.0) & (nu <= nu_ref1 + 40.0)
    x85_disp = nu[mask85_disp] - nu_ref1
    y85_disp = diff[mask85_disp]
    
    # Create Figure with exactly 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Subplot 1: 87Rb F=2 -> F'=3 ---
    ax = axes[0]
    ax.plot(x87_disp, y87_disp, 'k.', ms=3, alpha=0.45, label='Manifold Data')
    
    # Overplot fit on fit window
    xf87 = np.linspace(-span87, span87, 250)
    yf87 = lorentz_lin(xf87, *popt87)
    bg87 = popt87[3] + popt87[4] * xf87
    ax.plot(xf87, yf87, 'r-', lw=2.4, label='Isolated Lorentzian Fit')
    ax.plot(xf87, bg87, 'g--', lw=1.8, label='Fitted Baseline')
    
    # Vertical lines and shaded span for fitting window
    ax.axvline(-span87, color='purple', ls='--', lw=1.6, label=f'Fit Window ($\\pm${span87:.0f} MHz)')
    ax.axvline(span87, color='purple', ls='--', lw=1.6)
    ax.axvspan(-span87, span87, color='purple', alpha=0.08)
    
    info_87 = (
        f"FWHM ($\\Gamma$): {popt87[1]:.2f} $\\pm$ {perr87[1]:.2f} MHz\n"
        f"Amplitude: {popt87[2]:.2f} mV\n"
        f"Offset ($x_0$): {popt87[0]:.2f} MHz\n"
        f"Baseline $c_0$: {popt87[3]:.2f} mV\n"
        f"Baseline slope $c_1$: {popt87[4]:.4f} mV/MHz\n"
        f"RMS Residual: {rms87:.3f} mV"
    )
    ax.text(0.03, 0.96, info_87, transform=ax.transAxes, verticalalignment='top',
            fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.9))
    
    ax.set_title(f"$^{{87}}$Rb ($F=2 \\rightarrow F'=3$) Cycling Line in Full Manifold\nRun {r_id} — Power = {pwr:.2f} $\\mu$W",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Detuning from Cycling Peak (MHz)', fontsize=11)
    ax.set_ylabel('Subtracted Signal (mV)', fontsize=11)
    ax.set_xlim(-460, 40)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9.0)
    
    # --- Subplot 2: 85Rb F=3 -> F'=4 ---
    ax = axes[1]
    ax.plot(x85_disp, y85_disp, 'k.', ms=3, alpha=0.45, label='Manifold Data')
    
    # Overplot fit on fit window
    xf85 = np.linspace(-span85, span85, 250)
    yf85 = lorentz_lin(xf85, *popt85)
    bg85 = popt85[3] + popt85[4] * xf85
    ax.plot(xf85, yf85, 'b-', lw=2.4, label='Isolated Lorentzian Fit')
    ax.plot(xf85, bg85, 'g--', lw=1.8, label='Fitted Baseline')
    
    # Vertical lines and shaded span for fitting window
    ax.axvline(-span85, color='purple', ls='--', lw=1.6, label=f'Fit Window ($\\pm${span85:.0f} MHz)')
    ax.axvline(span85, color='purple', ls='--', lw=1.6)
    ax.axvspan(-span85, span85, color='purple', alpha=0.08)
    
    info_85 = (
        f"FWHM ($\\Gamma$): {popt85[1]:.2f} $\\pm$ {perr85[1]:.2f} MHz\n"
        f"Amplitude: {popt85[2]:.2f} mV\n"
        f"Offset ($x_0$): {popt85[0]:.2f} MHz\n"
        f"Baseline $c_0$: {popt85[3]:.2f} mV\n"
        f"Baseline slope $c_1$: {popt85[4]:.4f} mV/MHz\n"
        f"RMS Residual: {rms85:.3f} mV"
    )
    ax.text(0.03, 0.96, info_85, transform=ax.transAxes, verticalalignment='top',
            fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', alpha=0.9))
    
    ax.set_title(f"$^{{85}}$Rb ($F=3 \\rightarrow F'=4$) Cycling Line in Full Manifold\nRun {r_id} — Power = {pwr:.2f} $\\mu$W",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Detuning from Cycling Peak (MHz)', fontsize=11)
    ax.set_ylabel('Subtracted Signal (mV)', fontsize=11)
    ax.set_xlim(-220, 40)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9.0)
    
    plt.tight_layout()
    filename = f"power_{pwr:06.2f}uW_run{r_id:02d}.png"
    plt.savefig(os.path.join(out_dir_local, filename), dpi=200)
    plt.savefig(os.path.join(out_dir_brain, filename), dpi=200)
    if show:
        print(f"Showing interactive window for {filename} (Power: {pwr:6.2f} uW, 87Rb: {popt87[1]:.2f} MHz, 85Rb: {popt85[1]:.2f} MHz)...")
        plt.show()
    else:
        print(f"Saved: {filename} (Power: {pwr:6.2f} uW, 87Rb: {popt87[1]:.2f} MHz, 85Rb: {popt85[1]:.2f} MHz)")
    plt.close()

def main_cli():
    save_only = '--save-only' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--save-only']

    # If the user provides a specific run number (e.g. `python tests/plot_cycling_lines_per_power.py 10`)
    if len(args) > 0:
        try:
            target_run = int(args[0])
            if target_run in runs:
                target_power = powers[target_run - 1]
                print(f"Opening Run {target_run} ({target_power:.2f} uW)...")
                plot_power_run(target_run, target_power, show=not save_only)
                return
            else:
                print(f"Run {target_run} not found. Available runs: 1 to 22.")
        except ValueError:
            print("Usage: python tests/plot_cycling_lines_per_power.py [optional: run_number 1-22] [--save-only]")

    if not save_only:
        print(f"Opening interactive matplotlib figures for all 22 power runs (sorted by power)...")
        print(f"Close each figure window to proceed to the next power run, or press Ctrl+C in terminal to stop.")
    else:
        print(f"Saving all 22 power figures with wide manifold view and vlines...")

    for idx, (r_id, pwr) in enumerate(zip(sorted_runs, sorted_powers), 1):
        if not save_only:
            print(f"\n[{idx:02d}/22] Displaying Run {r_id} ({pwr:.2f} uW)...")
        plot_power_run(r_id, pwr, show=not save_only)
    print("\nFinished processing all runs!")

if __name__ == '__main__':
    main_cli()
