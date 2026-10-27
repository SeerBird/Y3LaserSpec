"""
show_s8_s9_local_cycling_fits.py
--------------------------------
Standalone interactive viewer and analyzer for the local Lorentzian + linear background
fits to the two cycling transitions:
  - 87Rb (F=2 -> F'=3), isolated window +/- 20 MHz
  - 85Rb (F=3 -> F'=4), isolated window +/- 18 MHz
across all 13 datasets from Session 8 (MAG1) and Session 9 (Protocols 1-7 and Magnetic Variation 1-5).

Run interactively (opens Matplotlib windows with plt.show()):
    python "plotting stuff/show_s8_s9_local_cycling_fits.py"
or inspect a specific dataset by index (1 to 13):
    python "plotting stuff/show_s8_s9_local_cycling_fits.py" 4
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import interp1d
import main

FSR = 374.74  # MHz / fringe

DATASETS = [
    ("S8: MAG1",          "Session 8: MAG1 (B ~ 2.5 G)",    "data/Session 8/FPMAG1.CSV", "data/Session 8/PMAG1.CSV", "data/Session 8/UMAG1.CSV", True,  (5.8, 7.3), (9.0, 10.3)),
    ("S9: Prot 1",        "Session 9: Protocol 1",          "data/Session 9/F1.CSV",     "data/Session 9/P1.CSV",    "data/Session 9/U1.CSV",    False, (3.7, 5.2), (7.4, 8.1)),
    ("S9: Prot 2",        "Session 9: Protocol 2",          "data/Session 9/F2.CSV",     "data/Session 9/P2.CSV",    "data/Session 9/U2.CSV",    False, (1.9, 3.3), (5.5, 6.2)),
    ("S9: Prot 3",        "Session 9: Protocol 3",          "data/Session 9/F3.CSV",     "data/Session 9/P3.CSV",    "data/Session 9/U3.CSV",    False, (1.9, 3.2), (5.5, 6.2)),
    ("S9: Prot 4",        "Session 9: Protocol 4",          "data/Session 9/F4.CSV",     "data/Session 9/P4.CSV",    "data/Session 9/U4.CSV",    False, (2.4, 3.8), (6.1, 6.8)),
    ("S9: Prot 5",        "Session 9: Protocol 5",          "data/Session 9/F5.CSV",     "data/Session 9/P5.CSV",    "data/Session 9/U5.CSV",    False, (3.3, 4.7), (7.0, 7.7)),
    ("S9: Prot 6",        "Session 9: Protocol 6",          "data/Session 9/F6.CSV",     "data/Session 9/P6.CSV",    "data/Session 9/U6.CSV",    False, (2.4, 3.7), (6.0, 6.7)),
    ("S9: Prot 7",        "Session 9: Protocol 7",          "data/Session 9/F7.CSV",     "data/Session 9/P7.CSV",    "data/Session 9/U7.CSV",    False, (2.7, 4.0), (6.3, 7.0)),
    ("S9: Mag 1 (19 mA)", "Session 9: Mag 1 (19 mA)",       "data/Session 9/Magnetic variation/GF1.CSV", "data/Session 9/Magnetic variation/GP1.CSV", "data/Session 9/Magnetic variation/GU1.CSV", False, (2.8, 4.6), (6.0, 7.6)),
    ("S9: Mag 2 (39 mA)", "Session 9: Mag 2 (39 mA)",       "data/Session 9/Magnetic variation/GF2.CSV", "data/Session 9/Magnetic variation/GP2.CSV", "data/Session 9/Magnetic variation/GU2.CSV", False, (2.8, 4.6), (6.0, 7.6)),
    ("S9: Mag 3 (59 mA)", "Session 9: Mag 3 (59 mA)",       "data/Session 9/Magnetic variation/GF3.CSV", "data/Session 9/Magnetic variation/GP3.CSV", "data/Session 9/Magnetic variation/GU3.CSV", False, (2.8, 4.6), (6.0, 7.6)),
    ("S9: Mag 4 (79 mA)", "Session 9: Mag 4 (79 mA)",       "data/Session 9/Magnetic variation/GF4.CSV", "data/Session 9/Magnetic variation/GP4.CSV", "data/Session 9/Magnetic variation/GU4.CSV", False, (2.8, 4.6), (6.0, 7.6)),
    ("S9: Mag 5 (98 mA)", "Session 9: Mag 5 (98 mA)",       "data/Session 9/Magnetic variation/GF5.CSV", "data/Session 9/Magnetic variation/GP5.CSV", "data/Session 9/Magnetic variation/GU5.CSV", False, (2.8, 4.6), (6.0, 7.6)),
]


def lorentz_lin(x, x0, gamma, A, c0, c1):
    """Lorentzian peak (FWHM = gamma) with a linear background c0 + c1*x."""
    return c0 + c1 * x + A / (1.0 + 4.0 * ((x - x0) / gamma) ** 2)


def prep_s8_s9_data(fp_name, pumped_name, unpumped_name, cut_edges=False, align_shift=True):
    _, fabry = main.load(fp_name)
    _, pumped = main.load(pumped_name)
    _, unpumped = main.load(unpumped_name)
    peaks, _ = find_peaks(fabry, prominence=1e-2, distance=100)
    index = np.arange(len(fabry))
    peak_index = np.arange(len(peaks))

    if cut_edges:
        linear_freq = np.poly1d(np.polyfit(peaks, peak_index, 1))
        edge_indices = find_peaks(peak_index - linear_freq(peaks))[0]
        if len(edge_indices) >= 2:
            e0 = edge_indices[0] + 1
            e1 = edge_indices[-1] - 1
            left_edge, right_edge = peaks[e0], peaks[e1]
            peaks = peaks[e0:e1 + 1] - left_edge
            peak_index = peak_index[e0:e1 + 1]
            fabry = fabry[left_edge:right_edge + 1]
            pumped = pumped[left_edge:right_edge + 1]
            unpumped = unpumped[left_edge:right_edge + 1]
            index = np.arange(len(fabry))

    shift_samples = 0.0
    if align_shift:
        u_interp = interp1d(index, unpumped, kind='cubic', bounds_error=False, fill_value=(unpumped[0], unpumped[-1]))
        pad = 200
        idx_eval = index[pad:-pad]
        p_eval = pumped[pad:-pad]
        bg_mask = np.abs(p_eval - unpumped[pad:-pad]) < 0.04
        if np.sum(bg_mask) < 1000:
            bg_mask = np.ones_like(p_eval, dtype=bool)
        res = minimize_scalar(
            lambda s: np.mean((p_eval[bg_mask] - u_interp(idx_eval[bg_mask] + s)) ** 2),
            bounds=(-300, 300),
            method='bounded'
        )
        shift_samples = float(res.x)
        unpumped = u_interp(index + shift_samples)

    freq_fit = np.poly1d(np.polyfit(peaks, peak_index, 5))
    freq = freq_fit(index)
    return freq, fabry, pumped, unpumped, shift_samples


def plot_dataset(idx_1based, show=True):
    short_label, full_label, fp_path, p_path, u_path, cut_edges, w87, w85 = DATASETS[idx_1based - 1]
    freq, _, p, u, shift_s = prep_s8_s9_data(fp_path, p_path, u_path, cut_edges=cut_edges, align_shift=True)
    diff = (p - u) * 1e3  # mV
    nu = freq * FSR

    # 87Rb F=2 -> F'=3 cycling peak
    m87 = (freq >= w87[0]) & (freq <= w87[1])
    nu_m87, y_m87 = nu[m87], diff[m87]
    pks87, _ = find_peaks(y_m87, prominence=15.0, distance=100)
    nu_ref87 = nu_m87[pks87[-1]]

    span87 = 20.0
    mask87_fit = (nu >= nu_ref87 - span87) & (nu <= nu_ref87 + span87)
    x87_fit = nu[mask87_fit] - nu_ref87
    y87_fit = diff[mask87_fit]
    p0_87 = [0.0, 11.0, np.max(y87_fit) - np.min(y87_fit), np.min(y87_fit), 0.0]
    popt87, pcov87 = curve_fit(
        lorentz_lin, x87_fit, y87_fit, p0=p0_87,
        bounds=([-8.0, 1.0, 0.0, -500.0, -10.0], [8.0, 50.0, 2000.0, 500.0, 10.0])
    )
    perr87 = np.sqrt(np.diag(pcov87))
    res87 = y87_fit - lorentz_lin(x87_fit, *popt87)
    rms87 = float(np.sqrt(np.mean(res87 ** 2)))
    cross87 = main.measure_fwhm_filtered_crossing(x87_fit, y87_fit, wing_margin=12.0)['fwhm']

    mask87_disp = (nu >= nu_ref87 - 460.0) & (nu <= nu_ref87 + 40.0)
    x87_disp = nu[mask87_disp] - nu_ref87
    y87_disp = diff[mask87_disp]

    # 85Rb F=3 -> F'=4 cycling peak
    m85 = (freq >= w85[0]) & (freq <= w85[1])
    nu_m85, y_m85 = nu[m85], diff[m85]
    pks85, _ = find_peaks(y_m85, prominence=15.0, distance=100)
    nu_ref85 = nu_m85[pks85[-1]]

    span85 = 18.0
    mask85_fit = (nu >= nu_ref85 - span85) & (nu <= nu_ref85 + span85)
    x85_fit = nu[mask85_fit] - nu_ref85
    y85_fit = diff[mask85_fit]
    p0_85 = [0.0, 11.0, np.max(y85_fit) - np.min(y85_fit), np.min(y85_fit), 0.0]
    popt85, pcov85 = curve_fit(
        lorentz_lin, x85_fit, y85_fit, p0=p0_85,
        bounds=([-8.0, 1.0, 0.0, -500.0, -10.0], [8.0, 50.0, 2500.0, 500.0, 10.0])
    )
    perr85 = np.sqrt(np.diag(pcov85))
    res85 = y85_fit - lorentz_lin(x85_fit, *popt85)
    rms85 = float(np.sqrt(np.mean(res85 ** 2)))
    cross85 = main.measure_fwhm_filtered_crossing(x85_fit, y85_fit, wing_margin=10.0)['fwhm']

    mask85_disp = (nu >= nu_ref85 - 220.0) & (nu <= nu_ref85 + 40.0)
    x85_disp = nu[mask85_disp] - nu_ref85
    y85_disp = diff[mask85_disp]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), gridspec_kw={'height_ratios': [2.2, 1.0], 'hspace': 0.32, 'wspace': 0.22})
    fig.canvas.manager.set_window_title(f"[{idx_1based}/13] {full_label} — Local Cycling Fits")

    # Top-left: 87Rb in manifold context
    ax = axes[0, 0]
    ax.plot(x87_disp, y87_disp, 'k.', ms=3, alpha=0.45, label='Manifold Data')
    xf87 = np.linspace(-span87, span87, 250)
    ax.plot(xf87, lorentz_lin(xf87, *popt87), 'r-', lw=2.2, label='Local Lorentzian Fit')
    ax.plot(xf87, popt87[3] + popt87[4] * xf87, 'g--', lw=1.8, label='Fitted Linear BG')
    ax.axvline(-span87, color='purple', ls='--', lw=1.5, label=f'Fit Window ($\\pm${span87:.0f} MHz)')
    ax.axvline(span87, color='purple', ls='--', lw=1.5)
    ax.axvspan(-span87, span87, color='purple', alpha=0.08)
    info87 = (
        f"Fitted FWHM ($\\Gamma$): {popt87[1]:.2f} $\\pm$ {perr87[1]:.2f} MHz\n"
        f"Filtered Crossing FWHM: {cross87:.2f} MHz\n"
        f"Amplitude: {popt87[2]:.1f} mV\n"
        f"Baseline $c_0$: {popt87[3]:.2f} mV, $c_1$: {popt87[4]:.3f} mV/MHz\n"
        f"RMS Residual: {rms87:.2f} mV ({rms87/popt87[2]*100:.1f}%)"
    )
    ax.text(0.03, 0.96, info87, transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='red', alpha=0.9))
    ax.set_title(f"$^{{87}}$Rb ($F=2 \\rightarrow F'=3$) — {full_label}", fontsize=11, fontweight='bold')
    ax.set_ylabel("Subtracted Signal (mV)", fontsize=10)
    ax.set_xlim(-460, 40)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5)

    # Bottom-left: 87Rb local fit residuals
    ax_res87 = axes[1, 0]
    ax_res87.plot(x87_fit, res87, 'r.-', ms=3, lw=1.0, alpha=0.85)
    ax_res87.axhline(0, color='k', ls='--', lw=0.9)
    ax_res87.set_title(f"$^{{87}}$Rb Local Fit Residuals (Window $\\pm{span87:.0f}$ MHz)", fontsize=10)
    ax_res87.set_xlabel("Detuning from Cycling Peak (MHz)", fontsize=10)
    ax_res87.set_ylabel("Data - Fit (mV)", fontsize=10)
    ax_res87.set_xlim(-span87, span87)
    ax_res87.grid(True, alpha=0.3)

    # Top-right: 85Rb in manifold context
    ax = axes[0, 1]
    ax.plot(x85_disp, y85_disp, 'k.', ms=3, alpha=0.45, label='Manifold Data')
    xf85 = np.linspace(-span85, span85, 250)
    ax.plot(xf85, lorentz_lin(xf85, *popt85), 'b-', lw=2.2, label='Local Lorentzian Fit')
    ax.plot(xf85, popt85[3] + popt85[4] * xf85, 'g--', lw=1.8, label='Fitted Linear BG')
    ax.axvline(-span85, color='purple', ls='--', lw=1.5, label=f'Fit Window ($\\pm${span85:.0f} MHz)')
    ax.axvline(span85, color='purple', ls='--', lw=1.5)
    ax.axvspan(-span85, span85, color='purple', alpha=0.08)
    info85 = (
        f"Fitted FWHM ($\\Gamma$): {popt85[1]:.2f} $\\pm$ {perr85[1]:.2f} MHz\n"
        f"Filtered Crossing FWHM: {cross85:.2f} MHz\n"
        f"Amplitude: {popt85[2]:.1f} mV\n"
        f"Baseline $c_0$: {popt85[3]:.2f} mV, $c_1$: {popt85[4]:.3f} mV/MHz\n"
        f"RMS Residual: {rms85:.2f} mV ({rms85/popt85[2]*100:.1f}%)"
    )
    ax.text(0.03, 0.96, info85, transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='blue', alpha=0.9))
    ax.set_title(f"$^{{85}}$Rb ($F=3 \\rightarrow F'=4$) — {full_label}", fontsize=11, fontweight='bold')
    ax.set_ylabel("Subtracted Signal (mV)", fontsize=10)
    ax.set_xlim(-220, 40)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5)

    # Bottom-right: 85Rb local fit residuals
    ax_res85 = axes[1, 1]
    ax_res85.plot(x85_fit, res85, 'b.-', ms=3, lw=1.0, alpha=0.85)
    ax_res85.axhline(0, color='k', ls='--', lw=0.9)
    ax_res85.set_title(f"$^{{85}}$Rb Local Fit Residuals (Window $\\pm{span85:.0f}$ MHz)", fontsize=10)
    ax_res85.set_xlabel("Detuning from Cycling Peak (MHz)", fontsize=10)
    ax_res85.set_ylabel("Data - Fit (mV)", fontsize=10)
    ax_res85.set_xlim(-span85, span85)
    ax_res85.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        print(f"Showing [{idx_1based:02d}/13] {full_label} | 87Rb: {popt87[1]:.2f} MHz (Cross: {cross87:.2f} MHz), 85Rb: {popt85[1]:.2f} MHz (Cross: {cross85:.2f} MHz)")
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        plot_dataset(idx, show=True)
    else:
        print("Cycling through all 13 datasets from Session 8 and Session 9 (close each window to advance)...")
        for i in range(1, len(DATASETS) + 1):
            plot_dataset(i, show=True)
