"""
plot_magnetic_quadratic_fits.py
--------------------------------
Fits the cycling transitions (87Rb F=2 -> F'=3 and 85Rb F=3 -> F'=4) across the
Session 9 Magnetic Variation datasets (currents: 19, 39, 59, 79, 98 mA) comparing:
  1. Isolated Lorentzian with Linear baseline (c0 + c1*x) - WITHOUT quadratic baseline
  2. Isolated Lorentzian with Quadratic baseline (c0 + c1*x + c2*x^2) - WITH quadratic baseline

Plots:
  - Interactive individual lineshape fits for each current with both models overlaid.
  - Overall Magnetic Field (Coil Current) vs. Linewidth relationship comparing
    Linear vs. Quadratic baseline fits on the same axes.

Usage:
  python "plotting stuff/plot_magnetic_quadratic_fits.py"           # Inspect runs interactively, then show summary
  python "plotting stuff/plot_magnetic_quadratic_fits.py" 39        # Inspect specific current (e.g. 39 mA)
  python "plotting stuff/plot_magnetic_quadratic_fits.py" --summary # Display only the overall summary plot
  python "plotting stuff/plot_magnetic_quadratic_fits.py" --save    # Batch-save all figures to figures/magnetic_variation/
"""

import sys
import os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import interp1d

out_dir = 'figures/magnetic_variation'
os.makedirs(out_dir, exist_ok=True)

data_dir = 'data/Session 9/Magnetic variation'
currents = [19, 39, 59, 79, 98]  # mA
fsr = 374.74  # MHz / fringe


def lorentz_lin(x, A, x0, gamma, c0, c1):
    return c0 + c1 * x + A / (1.0 + 4.0 * ((x - x0) / gamma) ** 2)


def lorentz_quad(x, A, x0, gamma, c0, c1, c2):
    return c0 + c1 * x + c2 * x ** 2 + A / (1.0 + 4.0 * ((x - x0) / gamma) ** 2)


def prep_magnetic_run(i):
    """Loads and calibrates a single magnetic variation dataset."""
    df_f = pd.read_csv(os.path.join(data_dir, f'GF{i}.CSV'))
    df_p = pd.read_csv(os.path.join(data_dir, f'GP{i}.CSV'))
    df_u = pd.read_csv(os.path.join(data_dir, f'GU{i}.CSV'))

    t = df_p.iloc[:, 0].to_numpy()
    p = df_p.iloc[:, 1].to_numpy()
    u = df_u.iloc[:, 1].to_numpy()
    f = df_f.iloc[:, 1].to_numpy()

    # Shift alignment between pumped and unpumped traces
    index = np.arange(len(p))
    u_interp = interp1d(index, u, kind='cubic', bounds_error=False, fill_value=(u[0], u[-1]))
    pad = 1000
    idx_eval = index[pad:-pad]
    p_eval = p[pad:-pad]
    bg_mask = np.abs(p_eval - u[pad:-pad]) < 0.04
    if np.sum(bg_mask) < 1000:
        bg_mask = np.ones_like(p_eval, dtype=bool)

    res = minimize_scalar(
        lambda s: np.mean((p_eval[bg_mask] - u_interp(idx_eval[bg_mask] + s)) ** 2),
        bounds=(-300, 300),
        method='bounded'
    )
    unpumped_aligned = u_interp(index + res.x)
    diff = (p - unpumped_aligned) * 1e3  # in mV

    # Fabry-Perot fringe detection & 5th-order polynomial frequency calibration
    peaks_fp, _ = find_peaks(f, prominence=0.01, distance=100)
    poly = np.poly1d(np.polyfit(peaks_fp, np.arange(len(peaks_fp)), 5))
    freq_fringe = poly(index)
    nu = freq_fringe * fsr

    return freq_fringe, nu, diff


def analyze_all_magnetic_runs(span87=25.0, span85=22.0):
    records = []
    print("Processing Session 9 Magnetic Variation runs (19, 39, 59, 79, 98 mA)...")
    for i, cur in enumerate(currents, 1):
        freq_fringe, nu, diff = prep_magnetic_run(i)

        # --- 87Rb F=2 -> F'=3 Cycling Transition (Manifold 1: fringe 2.8 to 4.6) ---
        m87_man = (freq_fringe >= 2.8) & (freq_fringe <= 4.6)
        x_m1 = nu[m87_man]
        y_m1 = diff[m87_man]
        pks87, _ = find_peaks(y_m1, prominence=15.0, distance=500)
        nu_ref87 = x_m1[pks87[-1]]

        m_win87 = np.abs(nu - nu_ref87) <= span87
        x87 = nu[m_win87] - nu_ref87
        y87 = diff[m_win87]

        # Linear fit 87
        p0_l87 = [y87.max() - y87.min(), 0.0, 11.0, y87.min(), 0.0]
        popt_l87, pcov_l87 = curve_fit(lorentz_lin, x87, y87, p0=p0_l87,
                                       bounds=([0, -5, 2, -200, -10], [500, 5, 50, 200, 10]))
        perr_l87 = np.sqrt(np.diag(pcov_l87))

        # Quadratic fit 87
        p0_q87 = [popt_l87[0], popt_l87[1], popt_l87[2], popt_l87[3], popt_l87[4], 0.0]
        popt_q87, pcov_q87 = curve_fit(lorentz_quad, x87, y87, p0=p0_q87,
                                       bounds=([0, -5, 2, -200, -10, -0.5], [500, 5, 50, 200, 10, 0.5]))
        perr_q87 = np.sqrt(np.diag(pcov_q87))

        # --- 85Rb F=3 -> F'=4 Cycling Transition (Manifold 2: fringe 6.0 to 7.6) ---
        m85_man = (freq_fringe >= 6.0) & (freq_fringe <= 7.6)
        x_m2 = nu[m85_man]
        y_m2 = diff[m85_man]
        pks85, _ = find_peaks(y_m2, prominence=15.0, distance=500)
        nu_ref85 = x_m2[pks85[-1]]

        m_win85 = np.abs(nu - nu_ref85) <= span85
        x85 = nu[m_win85] - nu_ref85
        y85 = diff[m_win85]

        # Linear fit 85
        p0_l85 = [y85.max() - y85.min(), 0.0, 10.0, y85.min(), 0.0]
        popt_l85, pcov_l85 = curve_fit(lorentz_lin, x85, y85, p0=p0_l85,
                                       bounds=([0, -5, 2, -200, -10], [500, 5, 50, 200, 10]))
        perr_l85 = np.sqrt(np.diag(pcov_l85))

        # Quadratic fit 85
        p0_q85 = [popt_l85[0], popt_l85[1], popt_l85[2], popt_l85[3], popt_l85[4], 0.0]
        popt_q85, pcov_q85 = curve_fit(lorentz_quad, x85, y85, p0=p0_q85,
                                       bounds=([0, -5, 2, -200, -10, -0.5], [500, 5, 50, 200, 10, 0.5]))
        perr_q85 = np.sqrt(np.diag(pcov_q85))

        records.append({
            'dataset_idx': i,
            'current': cur,
            # 87
            'x87': x87, 'y87': y87,
            'popt_l87': popt_l87, 'perr_l87': perr_l87,
            'popt_q87': popt_q87, 'perr_q87': perr_q87,
            'fwhm_l87': popt_l87[2], 'err_l87': perr_l87[2],
            'fwhm_q87': popt_q87[2], 'err_q87': perr_q87[2],
            'c2_87': popt_q87[5],
            # 85
            'x85': x85, 'y85': y85,
            'popt_l85': popt_l85, 'perr_l85': perr_l85,
            'popt_q85': popt_q85, 'perr_q85': perr_q85,
            'fwhm_l85': popt_l85[2], 'err_l85': perr_l85[2],
            'fwhm_q85': popt_q85[2], 'err_q85': perr_q85[2],
            'c2_85': popt_q85[5],
        })
    print("Magnetic variation processing complete.")
    return records


def plot_single_magnetic_run(rec, show=True, save_path=None):
    cur = rec['current']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    isotopes = [
        (axes[0], rec['x87'], rec['y87'], rec['popt_l87'], rec['perr_l87'],
         rec['popt_q87'], rec['perr_q87'], '87', "87Rb (F=2 -> F'=3) Cycling Peak"),
        (axes[1], rec['x85'], rec['y85'], rec['popt_l85'], rec['perr_l85'],
         rec['popt_q85'], rec['perr_q85'], '85', "85Rb (F=3 -> F'=4) Cycling Peak")
    ]

    for ax, x, y, pl, errl, pq, errq, iso, title in isotopes:
        ax.plot(x, y, '.', color='gray', alpha=0.35, ms=3.5, label='Measured Data')

        xf = np.linspace(x[0], x[-1], 300)
        yf_lin = lorentz_lin(xf, *pl)
        bg_lin = pl[3] + pl[4] * xf

        yf_quad = lorentz_quad(xf, *pq)
        bg_quad = pq[3] + pq[4] * xf + pq[5] * xf ** 2

        # Linear fit lines
        ax.plot(xf, bg_lin, '--', color='tab:gray', lw=1.6, label=f'Linear Base ($c_0={pl[3]:.1f}$ mV)')
        ax.plot(xf, yf_lin, ':', color='tab:orange' if iso == '87' else 'mediumpurple', lw=2.0,
                label=f'Linear Fit: $\\Gamma={pl[2]:.2f} \\pm {errl[2]:.2f}$ MHz')

        # Quadratic fit lines
        ax.plot(xf, bg_quad, '--', color='forestgreen', lw=1.8, label=f'Quadratic Base ($c_2={pq[5]:+.4f}$)')
        ax.plot(xf, yf_quad, '-', color='crimson' if iso == '87' else 'tab:blue', lw=2.4,
                label=f'Quadratic Fit: $\\Gamma={pq[2]:.2f} \\pm {errq[2]:.2f}$ MHz')

        info_box = (
            f"Solenoid Current: {cur} mA\n"
            f"Lin FWHM:  {pl[2]:.2f} $\\pm$ {errl[2]:.2f} MHz\n"
            f"Quad FWHM: {pq[2]:.2f} $\\pm$ {errq[2]:.2f} MHz\n"
            f"Curvature $c_2$: {pq[5]:+.4f} mV/MHz$^2$\n"
            f"Offset Difference: {pq[2] - pl[2]:+.2f} MHz"
        )
        ax.text(0.03, 0.96, info_box, transform=ax.transAxes, verticalalignment='top',
                fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                        edgecolor='crimson' if iso == '87' else 'tab:blue', alpha=0.9))

        ax.set_title(f"{title}\nCoil Current = {cur} mA", fontsize=11.5, fontweight='bold')
        ax.set_xlabel('Detuning from Peak Center (MHz)', fontsize=11)
        ax.set_ylabel('Photodiode Signal (mV)', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8.0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_summary_magnetic_relationship(records, show=True, save_path='figures/magnetic_variation/magnetic_field_vs_linewidth_comparison.png'):
    currents_arr = np.array([r['current'] for r in records])

    fl87 = np.array([r['fwhm_l87'] for r in records])
    el87 = np.array([r['err_l87'] for r in records])
    fq87 = np.array([r['fwhm_q87'] for r in records])
    eq87 = np.array([r['err_q87'] for r in records])

    fl85 = np.array([r['fwhm_l85'] for r in records])
    el85 = np.array([r['err_l85'] for r in records])
    fq85 = np.array([r['fwhm_q85'] for r in records])
    eq85 = np.array([r['err_q85'] for r in records])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel 1: 87Rb ---
    ax = axes[0]
    ax.axhline(6.065, color='darkgreen', ls='--', lw=2.0, alpha=0.85, label='Natural Linewidth $\\Gamma_{\\mathrm{nat}} = 6.07$ MHz')

    ax.errorbar(currents_arr, fl87, yerr=el87, fmt='o--', color='tab:orange', ms=7, capsize=4, lw=1.8,
                label='Without Quad Base (Linear Baseline)')
    ax.errorbar(currents_arr, fq87, yerr=eq87, fmt='s-', color='crimson', ms=7.5, capsize=4, lw=2.2,
                label='With Quad Base (Quadratic Baseline)')

    # Highlight minimum
    min_idx87 = np.argmin(fq87)
    ax.annotate(f"Minimum at {currents_arr[min_idx87]} mA\nLin:  {fl87[min_idx87]:.2f} MHz\nQuad: {fq87[min_idx87]:.2f} MHz",
                xy=(currents_arr[min_idx87], fq87[min_idx87]),
                xytext=(currents_arr[min_idx87] + 8, fq87[min_idx87] - 2.5),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5),
                fontsize=9.5, fontweight='bold', color='crimson',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='crimson', alpha=0.9))

    ax.set_title("$^{87}$Rb ($F=2 \\rightarrow F'=3$) Linewidth vs. Solenoid Current\nLinear vs. Quadratic Baseline Comparison", fontsize=12, fontweight='bold')
    ax.set_xlabel('Coil Current (mA)', fontsize=11)
    ax.set_ylabel('Fitted FWHM (MHz)', fontsize=11)
    ax.set_xlim(10, 105)
    ax.set_ylim(8.0, 18.0)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9.0, framealpha=0.95)

    # --- Panel 2: 85Rb ---
    ax = axes[1]
    ax.axhline(6.065, color='darkgreen', ls='--', lw=2.0, alpha=0.85, label='Natural Linewidth $\\Gamma_{\\mathrm{nat}} = 6.07$ MHz')

    ax.errorbar(currents_arr, fl85, yerr=el85, fmt='o--', color='mediumpurple', ms=7, capsize=4, lw=1.8,
                label='Without Quad Base (Linear Baseline)')
    ax.errorbar(currents_arr, fq85, yerr=eq85, fmt='s-', color='tab:blue', ms=7.5, capsize=4, lw=2.2,
                label='With Quad Base (Quadratic Baseline)')

    min_idx85 = np.argmin(fq85)
    ax.annotate(f"Minimum at {currents_arr[min_idx85]} mA\nLin:  {fl85[min_idx85]:.2f} MHz\nQuad: {fq85[min_idx85]:.2f} MHz",
                xy=(currents_arr[min_idx85], fq85[min_idx85]),
                xytext=(currents_arr[min_idx85] + 8, fq85[min_idx85] - 2.5),
                arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1.5),
                fontsize=9.5, fontweight='bold', color='tab:blue',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='tab:blue', alpha=0.9))

    ax.set_title("$^{85}$Rb ($F=3 \\rightarrow F'=4$) Linewidth vs. Solenoid Current\nLinear vs. Quadratic Baseline Comparison", fontsize=12, fontweight='bold')
    ax.set_xlabel('Coil Current (mA)', fontsize=11)
    ax.set_ylabel('Fitted FWHM (MHz)', fontsize=11)
    ax.set_xlim(10, 105)
    ax.set_ylim(8.0, 20.0)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9.0, framealpha=0.95)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved summary figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    records = analyze_all_magnetic_runs()

    # Save metrics table
    df_out = pd.DataFrame([{
        'current_mA': r['current'],
        'fwhm_lin_87': r['fwhm_l87'], 'err_lin_87': r['err_l87'],
        'fwhm_quad_87': r['fwhm_q87'], 'err_quad_87': r['err_q87'], 'c2_87': r['c2_87'],
        'fwhm_lin_85': r['fwhm_l85'], 'err_lin_85': r['err_l85'],
        'fwhm_quad_85': r['fwhm_q85'], 'err_quad_85': r['err_q85'], 'c2_85': r['c2_85'],
    } for r in records])
    csv_path = 'figures/magnetic_variation/magnetic_quadratic_vs_linear_summary.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"Saved summary metrics to: {csv_path}")

    args = sys.argv[1:]
    if '--save' in args:
        print("Saving all 5 individual current figures...")
        for rec in records:
            p_save = os.path.join(out_dir, f"magnetic_run_{rec['current']:02d}mA.png")
            plot_single_magnetic_run(rec, show=False, save_path=p_save)
        plot_summary_magnetic_relationship(records, show=False)
        print("All figures saved.")
        return

    if '--summary' in args:
        plot_summary_magnetic_relationship(records, show=True)
        return

    # Check for specific current argument (e.g. `python ... 39`)
    cur_arg = None
    for a in args:
        try:
            val = int(a)
            if val in currents:
                cur_arg = val
                break
        except ValueError:
            pass

    if cur_arg is not None:
        rec = next(r for r in records if r['current'] == cur_arg)
        print(f"Displaying Current {cur_arg} mA...")
        plot_single_magnetic_run(rec, show=True)
        return

    # Default: loop through all 5 currents, then show summary plot
    print("\n[Interactive Mode: Displaying all 5 magnetic variation currents (19, 39, 59, 79, 98 mA)]")
    print("Close each window to proceed to the next current.")
    for idx, rec in enumerate(records, 1):
        print(f"[{idx}/5] Current {rec['current']:02d} mA — "
              f"87Rb Lin: {rec['fwhm_l87']:.2f} -> Quad: {rec['fwhm_q87']:.2f} MHz")
        plot_single_magnetic_run(rec, show=True)

    print("\nDisplaying overall Magnetic Field vs. Linewidth comparison...")
    plot_summary_magnetic_relationship(records, show=True)


if __name__ == '__main__':
    main()
