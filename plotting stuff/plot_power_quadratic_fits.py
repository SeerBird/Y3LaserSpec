"""
plot_power_quadratic_fits.py
----------------------------
Fits the cycling transitions (87Rb F=2 -> F'=3 and 85Rb F=3 -> F'=4) across all
22 power variation runs (Session 10) comparing:
  1. Lorentzian fit with standard Linear baseline (c0 + c1*x)
  2. Lorentzian fit with Quadratic baseline (c0 + c1*x + c2*x^2) to accommodate
     the upward-curving algebraic wing of the nearby crossover resonance.

Plots:
  - Interactive individual line fits for each power run with both models overlaid.
  - Overall Power vs. Linewidth relationship comparing Linear vs. Quadratic baseline
    extrapolations to the theoretical natural linewidth Gamma_nat = 6.07 MHz.

Usage:
  python "plotting stuff/plot_power_quadratic_fits.py"           # Inspect runs interactively, then show summary
  python "plotting stuff/plot_power_quadratic_fits.py" 10        # Inspect specific Run (e.g. Run 10 at 19.6 uW)
  python "plotting stuff/plot_power_quadratic_fits.py" --summary # Display only the overall summary plot
  python "plotting stuff/plot_power_quadratic_fits.py" --save    # Batch-save all figures to figures/power_quadratic_fits/
"""

import sys
import os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from main import load_and_calibrate, select_cycling_peak

out_dir = 'figures/power_quadratic_fits'
os.makedirs(out_dir, exist_ok=True)

# Load power variation metadata
powers_df = pd.read_excel('data/Session 10/Power variation/Powers.xlsx')
powers_array = powers_df['Power_uW'].values
all_runs = np.arange(1, len(powers_array) + 1)


def lorentz_lin(x, A, x0, gamma, c0, c1):
    """Lorentzian with linear baseline."""
    return c0 + c1 * x + A / (1.0 + 4.0 * ((x - x0) / gamma) ** 2)


def lorentz_quad(x, A, x0, gamma, c0, c1, c2):
    """Lorentzian with quadratic baseline."""
    return c0 + c1 * x + c2 * x ** 2 + A / (1.0 + 4.0 * ((x - x0) / gamma) ** 2)


def power_broadening_model(p, gamma_0, p_sat):
    return gamma_0 * np.sqrt(1.0 + p / p_sat)


def analyze_all_power_runs(span87=25.0, span85=22.0):
    records = []
    print("Fitting all 22 power runs with Linear vs. Quadratic baselines...")
    for r in all_runs:
        pwr = powers_array[r - 1]
        p_path = f'data/Session 10/Power variation/P{r}.CSV'
        u_path = f'data/Session 10/Power variation/U{r}.CSV'
        f_path = f'data/Session 10/Power variation/F{r}.CSV'
        if not os.path.exists(f_path):
            f_path = 'data/Session 10/Power variation/F14.CSV'

        t, nu, diff = load_and_calibrate(f_path, p_path, u_path)

        # --- 87Rb F=2 -> F'=3 ---
        pk87 = select_cycling_peak(t, nu, diff, isotope='87', span_mhz=span87)
        x87, y87 = pk87['x'], pk87['y']
        
        # Linear fit 87
        p0_l87 = [y87.max() - y87.min(), 0.0, 7.0, y87.min(), 0.0]
        popt_l87, pcov_l87 = curve_fit(lorentz_lin, x87, y87, p0=p0_l87,
                                       bounds=([0, -5, 1, -15, -1], [100, 5, 40, 15, 1]))
        perr_l87 = np.sqrt(np.diag(pcov_l87))

        # Quadratic fit 87
        p0_q87 = [popt_l87[0], popt_l87[1], popt_l87[2], popt_l87[3], popt_l87[4], 0.0]
        popt_q87, pcov_q87 = curve_fit(lorentz_quad, x87, y87, p0=p0_q87,
                                       bounds=([0, -5, 1, -15, -1, -0.05], [100, 5, 40, 15, 1, 0.05]))
        perr_q87 = np.sqrt(np.diag(pcov_q87))

        # --- 85Rb F=3 -> F'=4 ---
        pk85 = select_cycling_peak(t, nu, diff, isotope='85', span_mhz=span85)
        x85, y85 = pk85['x'], pk85['y']

        # Linear fit 85
        p0_l85 = [y85.max() - y85.min(), 0.0, 7.0, y85.min(), 0.0]
        popt_l85, pcov_l85 = curve_fit(lorentz_lin, x85, y85, p0=p0_l85,
                                       bounds=([0, -5, 1, -15, -1], [100, 5, 40, 15, 1]))
        perr_l85 = np.sqrt(np.diag(pcov_l85))

        # Quadratic fit 85
        p0_q85 = [popt_l85[0], popt_l85[1], popt_l85[2], popt_l85[3], popt_l85[4], 0.0]
        popt_q85, pcov_q85 = curve_fit(lorentz_quad, x85, y85, p0=p0_q85,
                                       bounds=([0, -5, 1, -15, -1, -0.05], [100, 5, 40, 15, 1, 0.05]))
        perr_q85 = np.sqrt(np.diag(pcov_q85))

        records.append({
            'run': r,
            'power': pwr,
            # 87 data
            'x87': x87, 'y87': y87,
            'popt_l87': popt_l87, 'perr_l87': perr_l87,
            'popt_q87': popt_q87, 'perr_q87': perr_q87,
            'fwhm_l87': popt_l87[2], 'fwhm_q87': popt_q87[2],
            'c2_87': popt_q87[5],
            # 85 data
            'x85': x85, 'y85': y85,
            'popt_l85': popt_l85, 'perr_l85': perr_l85,
            'popt_q85': popt_q85, 'perr_q85': perr_q85,
            'fwhm_l85': popt_l85[2], 'fwhm_q85': popt_q85[2],
            'c2_85': popt_q85[5],
        })
    print("Fitting complete.")
    return records


def plot_single_run(rec, show=True, save_path=None):
    r_id = rec['run']
    pwr = rec['power']

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
        ax.plot(xf, bg_lin, '--', color='tab:gray', lw=1.6, label=f'Linear Base ($c_0={pl[3]:.2f}$)')
        ax.plot(xf, yf_lin, ':', color='tab:orange' if iso == '87' else 'mediumpurple', lw=2.0,
                label=f'Linear Fit: $\\Gamma={pl[2]:.2f} \\pm {errl[2]:.2f}$ MHz')

        # Quadratic fit lines
        ax.plot(xf, bg_quad, '--', color='forestgreen', lw=1.8, label=f'Quadratic Base ($c_2={pq[5]:+.5f}$)')
        ax.plot(xf, yf_quad, '-', color='crimson' if iso == '87' else 'tab:blue', lw=2.4,
                label=f'Quadratic Fit: $\\Gamma={pq[2]:.2f} \\pm {errq[2]:.2f}$ MHz')

        info_box = (
            f"Run {r_id} — Power: {pwr:.2f} $\\mu$W\n"
            f"Lin FWHM:  {pl[2]:.2f} $\\pm$ {errl[2]:.2f} MHz\n"
            f"Quad FWHM: {pq[2]:.2f} $\\pm$ {errq[2]:.2f} MHz\n"
            f"Curvature $c_2$: {pq[5]:+.5f} mV/MHz$^2$\n"
            f"Difference: {pq[2] - pl[2]:+.2f} MHz"
        )
        ax.text(0.03, 0.96, info_box, transform=ax.transAxes, verticalalignment='top',
                fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                        edgecolor='crimson' if iso == '87' else 'tab:blue', alpha=0.9))

        ax.set_title(f"{title}\nRun {r_id} (Power = {pwr:.2f} $\\mu$W)", fontsize=11.5, fontweight='bold')
        ax.set_xlabel('Detuning from Peak Center (MHz)', fontsize=11)
        ax.set_ylabel('Signal (mV)', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8.0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_summary_power_relationship(records, show=True, save_path='figures/power_quadratic_vs_linear_summary.png'):
    powers = np.array([r['power'] for r in records])
    fl87 = np.array([r['fwhm_l87'] for r in records])
    fq87 = np.array([r['fwhm_q87'] for r in records])
    fl85 = np.array([r['fwhm_l85'] for r in records])
    fq85 = np.array([r['fwhm_q85'] for r in records])

    # Fits for 87Rb
    popt_l87, pcov_l87 = curve_fit(power_broadening_model, powers, fl87, p0=[5.0, 500.0])
    perr_l87 = np.sqrt(np.diag(pcov_l87))
    popt_q87, pcov_q87 = curve_fit(power_broadening_model, powers, fq87, p0=[5.5, 550.0])
    perr_q87 = np.sqrt(np.diag(pcov_q87))

    # Fits for 85Rb (reliable powers >= 35 uW)
    m85 = powers >= 35.0
    popt_l85, pcov_l85 = curve_fit(power_broadening_model, powers[m85], fl85[m85], p0=[4.0, 400.0])
    perr_l85 = np.sqrt(np.diag(pcov_l85))
    popt_q85, pcov_q85 = curve_fit(power_broadening_model, powers[m85], fq85[m85], p0=[5.0, 450.0])
    perr_q85 = np.sqrt(np.diag(pcov_q85))

    p_dense = np.linspace(0, np.max(powers) * 1.05, 300)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # --- Panel 1: 87Rb ---
    ax = axes[0]
    ax.axhline(6.065, color='darkgreen', ls='--', lw=2.0, alpha=0.85, label='Natural Linewidth $\\Gamma_{\\mathrm{nat}} = 6.07$ MHz')
    ax.plot(powers, fl87, 'o', color='tab:orange', ms=6.5, mec='black', mew=0.8, alpha=0.8, label='87Rb Data (Linear Base)')
    ax.plot(p_dense, power_broadening_model(p_dense, *popt_l87), '--', color='tab:orange', lw=2.0,
            label=f'Linear Base Fit: $\\Gamma_0 = {popt_l87[0]:.2f} \\pm {perr_l87[0]:.2f}$ MHz, $P_{{\\mathrm{{sat}}}} = {popt_l87[1]:.0f}$ $\\mu$W')

    ax.plot(powers, fq87, 's', color='crimson', ms=7.0, mec='black', mew=1.0, alpha=0.9, label='87Rb Data (Quadratic Base)')
    ax.plot(p_dense, power_broadening_model(p_dense, *popt_q87), '-', color='crimson', lw=2.4,
            label=f'Quadratic Base Fit: $\\Gamma_0 = {popt_q87[0]:.2f} \\pm {perr_q87[0]:.2f}$ MHz, $P_{{\\mathrm{{sat}}}} = {popt_q87[1]:.0f}$ $\\mu$W')

    ax.set_title("$^{87}$Rb ($F=2 \\rightarrow F'=3$) Linewidth vs. Beam Power\nLinear vs. Quadratic Baseline Comparison", fontsize=12, fontweight='bold')
    ax.set_xlabel('Beam Power ($\\mu$W)', fontsize=11)
    ax.set_ylabel('Fitted FWHM (MHz)', fontsize=11)
    ax.set_xlim(-20, np.max(powers) * 1.05)
    ax.set_ylim(4.0, 10.5)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=8.8, framealpha=0.95)

    # --- Panel 2: 85Rb ---
    ax = axes[1]
    ax.axhline(6.065, color='darkgreen', ls='--', lw=2.0, alpha=0.85, label='Natural Linewidth $\\Gamma_{\\mathrm{nat}} = 6.07$ MHz')
    ax.plot(powers[m85], fl85[m85], 'o', color='mediumpurple', ms=6.5, mec='black', mew=0.8, alpha=0.8, label='85Rb Data (Linear Base)')
    ax.plot(p_dense, power_broadening_model(p_dense, *popt_l85), '--', color='mediumpurple', lw=2.0,
            label=f'Linear Base Fit: $\\Gamma_0 = {popt_l85[0]:.2f} \\pm {perr_l85[0]:.2f}$ MHz, $P_{{\\mathrm{{sat}}}} = {popt_l85[1]:.0f}$ $\\mu$W')

    ax.plot(powers[m85], fq85[m85], 's', color='tab:blue', ms=7.0, mec='black', mew=1.0, alpha=0.9, label='85Rb Data (Quadratic Base)')
    ax.plot(p_dense, power_broadening_model(p_dense, *popt_q85), '-', color='tab:blue', lw=2.4,
            label=f'Quadratic Base Fit: $\\Gamma_0 = {popt_q85[0]:.2f} \\pm {perr_q85[0]:.2f}$ MHz, $P_{{\\mathrm{{sat}}}} = {popt_q85[1]:.0f}$ $\\mu$W')

    ax.set_title("$^{85}$Rb ($F=3 \\rightarrow F'=4$) Linewidth vs. Beam Power\nLinear vs. Quadratic Baseline Comparison", fontsize=12, fontweight='bold')
    ax.set_xlabel('Beam Power ($\\mu$W)', fontsize=11)
    ax.set_ylabel('Fitted FWHM (MHz)', fontsize=11)
    ax.set_xlim(-20, np.max(powers) * 1.05)
    ax.set_ylim(3.5, 10.5)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=8.8, framealpha=0.95)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved summary figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    records = analyze_all_power_runs()

    # Save metrics table
    df_out = pd.DataFrame([{
        'run': r['run'], 'power_uW': r['power'],
        'fwhm_lin_87': r['fwhm_l87'], 'fwhm_quad_87': r['fwhm_q87'], 'c2_87': r['c2_87'],
        'fwhm_lin_85': r['fwhm_l85'], 'fwhm_quad_85': r['fwhm_q85'], 'c2_85': r['c2_85'],
    } for r in records])
    csv_path = 'figures/power_quadratic_vs_linear_summary.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"Saved summary metrics to: {csv_path}")

    args = sys.argv[1:]
    if '--save' in args:
        print("Saving all 22 interactive run figures...")
        for rec in records:
            p_save = os.path.join(out_dir, f"power_run_{rec['run']:02d}_{rec['power']:06.2f}uW.png")
            plot_single_run(rec, show=False, save_path=p_save)
        plot_summary_power_relationship(records, show=False)
        print("All figures saved.")
        return

    if '--summary' in args:
        plot_summary_power_relationship(records, show=True)
        return

    # Check for specific run argument (e.g. `python ... 10`)
    run_arg = None
    for a in args:
        try:
            val = int(a)
            if 1 <= val <= 22:
                run_arg = val
                break
        except ValueError:
            pass

    if run_arg is not None:
        rec = next(r for r in records if r['run'] == run_arg)
        print(f"Displaying Run {run_arg} ({rec['power']:.2f} uW)...")
        plot_single_run(rec, show=True)
        return

    # Default: loop through all 22 runs in order of increasing power, then show summary
    sorted_recs = sorted(records, key=lambda x: x['power'])
    total = len(sorted_recs)
    print(f"\n[Interactive Mode: Displaying all {total} runs in order of increasing power]")
    print("Close each window to proceed to the next run.")
    for idx, rec in enumerate(sorted_recs, 1):
        print(f"[{idx:02d}/{total}] Run {rec['run']:02d} ({rec['power']:6.2f} uW) — "
              f"87Rb Lin: {rec['fwhm_l87']:.2f} -> Quad: {rec['fwhm_q87']:.2f} MHz")
        plot_single_run(rec, show=True)

    print("\nDisplaying overall Power vs. Linewidth comparison...")
    plot_summary_power_relationship(records, show=True)


if __name__ == '__main__':
    main()
