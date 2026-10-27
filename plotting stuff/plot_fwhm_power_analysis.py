"""
plot_fwhm_power_analysis.py
----------------------------
Plots:
1. The FWHM vs. Power relationship across all 22 power variation runs for both
   cycling lines (87Rb F=2 -> F'=3 and 85Rb F=3 -> F'=4), with theoretical
   power-broadening fits: Gamma(P) = Gamma_0 * sqrt(1 + P / P_sat).
2. Interactive step-by-step visual displays of how the noise-filtered crossing
   method extracts the FWHM for each individual peak (raw data, baseline fit,
   Savitzky-Golay smoothed contour, half-maximum threshold level, and sub-sample
   interpolated crossing points).

Usage:
  python "plotting stuff/plot_fwhm_power_analysis.py"            # Plots FWHM vs Power summary & shows sample run 10
  python "plotting stuff/plot_fwhm_power_analysis.py" 10         # Interactively inspect Run 10 peak acquisition
  python "plotting stuff/plot_fwhm_power_analysis.py" --all      # Interactively inspect all 22 runs in sequence
  python "plotting stuff/plot_fwhm_power_analysis.py" --save     # Save all figures to figures/fwhm_acquisition/
"""

import sys
import os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from main import load_and_calibrate, select_cycling_peak, measure_fwhm_filtered_crossing

# Setup directories
out_dir_runs = 'figures/fwhm_acquisition'
os.makedirs(out_dir_runs, exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Load power variation metadata
powers_path = 'data/Session 10/Power variation/Powers.xlsx'
df_powers = pd.read_excel(powers_path)
powers_array = df_powers['Power_uW'].values
all_run_ids = np.arange(1, len(powers_array) + 1)


def analyze_all_power_runs(span87=20.0, span85=18.0, wing87=12.0, wing85=10.0):
    """
    Processes all 22 power runs, extracting FWHM and intermediate data
    for both 87Rb and 85Rb cycling transitions.
    """
    records = []
    print("Extracting cycling lines across all 22 power runs...")
    for r in all_run_ids:
        pwr = powers_array[r - 1]
        p_path = f'data/Session 10/Power variation/P{r}.CSV'
        u_path = f'data/Session 10/Power variation/U{r}.CSV'
        f_path = f'data/Session 10/Power variation/F{r}.CSV'
        if not os.path.exists(f_path):
            f_path = 'data/Session 10/Power variation/F14.CSV'

        t, nu, diff = load_and_calibrate(f_path, p_path, u_path, fsr=374.74)

        # 87Rb F=2 -> F'=3 cycling peak
        peak87 = select_cycling_peak(t, nu, diff, isotope='87', span_mhz=span87)
        res87 = measure_fwhm_filtered_crossing(peak87['x'], peak87['y'], wing_margin=wing87)

        # 85Rb F=3 -> F'=4 cycling peak
        peak85 = select_cycling_peak(t, nu, diff, isotope='85', span_mhz=span85)
        res85 = measure_fwhm_filtered_crossing(peak85['x'], peak85['y'], wing_margin=wing85)

        records.append({
            'run': r,
            'power_uW': pwr,
            'fwhm_87': res87['fwhm'],
            'amp_87': res87['y_max'],
            'res87': res87,
            'peak87': peak87,
            'fwhm_85': res85['fwhm'],
            'amp_85': res85['y_max'],
            'res85': res85,
            'peak85': peak85,
        })
    print("Extraction complete.")
    return records


def power_broadening_model(p, gamma_0, p_sat):
    """Theoretical saturation power-broadened linewidth: Gamma(P) = Gamma_0 * sqrt(1 + P / P_sat)."""
    return gamma_0 * np.sqrt(1.0 + p / p_sat)


def plot_single_run_acquisition(rec, show=True, save_path=None):
    """
    Generates an interactive 2-panel figure showing each intermediate step
    of the noise-filtered crossing FWHM acquisition for a single run:
      - Raw data points
      - Wing baseline regions and fitted baseline y_bg(x)
      - Savitzky-Golay smoothed profile
      - Half-maximum threshold level
      - Sub-sample interpolated crossing points
      - Final FWHM span
    """
    r_id = rec['run']
    pwr = rec['power_uW']
    res87 = rec['res87']
    res85 = rec['res85']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    isotopes_data = [
        (axes[0], res87, '87', 'crimson', 12.0, "87Rb (F=2 -> F'=3) Cycling Peak"),
        (axes[1], res85, '85', 'tab:blue', 10.0, "85Rb (F=3 -> F'=4) Cycling Peak")
    ]

    for ax, res, iso, color, wing_m, title in isotopes_data:
        x = res['x']
        y = res['y']
        y_bg = res['y_bg']
        y_smooth_net = res['y_smooth']
        y_smooth_total = y_smooth_net + y_bg
        x_left = res['x_left']
        x_right = res['x_right']
        hm_net = res['hm_level']
        hm_left_total = hm_net + (res['c0'] + res['c1'] * x_left)
        hm_right_total = hm_net + (res['c0'] + res['c1'] * x_right)
        fwhm = res['fwhm']
        x_max = res['x_max']
        y_max_total = res['y_max'] + (res['c0'] + res['c1'] * x_max)

        # Step 1: Raw differential data
        ax.plot(x, y, '.', color='gray', alpha=0.35, ms=3.5, label='Raw Differential Data $(y_i)$')

        # Step 2: Highlight baseline fitting wings
        ax.axvspan(x[0], -wing_m, color='gray', alpha=0.10, label=f'Baseline Wing Fit ($|x| \\geq {wing_m:.0f}$ MHz)')
        ax.axvspan(wing_m, x[-1], color='gray', alpha=0.10)

        # Step 3: Fitted baseline
        ax.plot(x, y_bg, '--', color='darkslategray', lw=1.8,
                label=f'Linear Baseline: $c_0={res["c0"]:.2f}$ mV, $c_1={res["c1"]:.3f}$ mV/MHz')

        # Step 4: Savitzky-Golay filtered contour
        ax.plot(x, y_smooth_total, '-', color=color, lw=2.4,
                label='Noise-Filtered Contour (Savitzky-Golay, $W=31$ pts)')

        # Step 5: Peak Summit
        ax.plot(x_max, y_max_total, 'v', color='darkgreen', ms=8,
                label=f'Peak Apex ($A = {res["y_max"]:.2f}$ mV)')

        # Step 6: Half-maximum threshold level line
        ax.plot([x_left, x_right], [hm_left_total, hm_right_total], '--', color='green', lw=2.0,
                label=f'Half-Maximum Level ($50\\%$ Amplitude)')

        # Step 7: Sub-sample crossing points
        ax.plot(x_left, hm_left_total, 'o', color='purple', ms=8, mec='black', mew=1.2,
                label=f'Crossings: $\\nu_L={x_left:+.2f}$, $\\nu_R={x_right:+.2f}$ MHz')
        ax.plot(x_right, hm_right_total, 'o', color='purple', ms=8, mec='black', mew=1.2)

        # Drop lines to baseline
        bg_left = res['c0'] + res['c1'] * x_left
        bg_right = res['c0'] + res['c1'] * x_right
        ax.plot([x_left, x_left], [bg_left, hm_left_total], ':', color='purple', lw=1.5)
        ax.plot([x_right, x_right], [bg_right, hm_right_total], ':', color='purple', lw=1.5)

        # Step 8: Double-headed arrow indicating FWHM
        y_arrow = 0.5 * (hm_left_total + hm_right_total)
        ax.annotate('', xy=(x_right, y_arrow), xytext=(x_left, y_arrow),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.8))
        ax.text(0.5 * (x_left + x_right), y_arrow + 0.12 * res['y_max'],
                f'FWHM = {fwhm:.2f} MHz',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='yellow', alpha=0.85, edgecolor='black'))

        # Metrics summary box
        info_text = (
            f"Run {r_id} — Power: {pwr:.2f} $\\mu$W\n"
            f"FWHM: {fwhm:.2f} MHz\n"
            f"Amplitude: {res['y_max']:.2f} mV\n"
            f"Baseline $c_0$: {res['c0']:.2f} mV\n"
            f"Left Crossing: {x_left:+.2f} MHz\n"
            f"Right Crossing: {x_right:+.2f} MHz\n"
            f"Filter: SavGol (31 pts, poly 3)"
        )
        ax.text(0.03, 0.96, info_text, transform=ax.transAxes, verticalalignment='top',
                fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, alpha=0.9))

        ax.set_title(f"{title}\nRun {r_id} (Power = {pwr:.2f} $\\mu$W)", fontsize=11.5, fontweight='bold')
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


def plot_fwhm_vs_power(records, show=True, save_path='figures/fwhm_power_relationship.png'):
    """
    Plots the FWHM as a function of pump-probe beam power for both 87Rb and 85Rb cycling lines,
    comparing:
      1. Full-dataset power-broadening fit
      2. High-power regime fit (P >= 200 uW)
    fitting the saturation power-broadening model Gamma(P) = Gamma_0 * sqrt(1 + P / P_sat).
    """
    powers = np.array([r['power_uW'] for r in records])
    f87 = np.array([r['fwhm_87'] for r in records])
    f85 = np.array([r['fwhm_85'] for r in records])

    # 1. Fit 87Rb across all power runs
    popt87_all, pcov87_all = curve_fit(power_broadening_model, powers, f87, p0=[6.0, 600.0],
                                       bounds=([1.0, 10.0], [15.0, 5000.0]))
    perr87_all = np.sqrt(np.diag(pcov87_all))

    # 2. Fit 87Rb high-power regime only (P >= 200 uW)
    m87_hp = powers >= 200.0
    popt87_hp, pcov87_hp = curve_fit(power_broadening_model, powers[m87_hp], f87[m87_hp], p0=[6.0, 800.0],
                                     bounds=([1.0, 10.0], [15.0, 5000.0]))
    perr87_hp = np.sqrt(np.diag(pcov87_hp))

    # 3. Fit 85Rb all reliable data (P >= 35 uW)
    mask85_all = powers >= 35.0
    popt85_all, pcov85_all = curve_fit(power_broadening_model, powers[mask85_all], f85[mask85_all], p0=[6.0, 600.0],
                                       bounds=([1.0, 10.0], [15.0, 5000.0]))
    perr85_all = np.sqrt(np.diag(pcov85_all))

    # 4. Fit 85Rb high-power regime only (P >= 200 uW)
    m85_hp = powers >= 200.0
    popt85_hp, pcov85_hp = curve_fit(power_broadening_model, powers[m85_hp], f85[m85_hp], p0=[6.0, 800.0],
                                     bounds=([1.0, 10.0], [15.0, 5000.0]))
    perr85_hp = np.sqrt(np.diag(pcov85_hp))

    p_dense = np.linspace(0, np.max(powers) * 1.05, 300)
    fit87_all_dense = power_broadening_model(p_dense, *popt87_all)
    fit87_hp_dense = power_broadening_model(p_dense, *popt87_hp)
    fit85_all_dense = power_broadening_model(p_dense, *popt85_all)
    fit85_hp_dense = power_broadening_model(p_dense, *popt85_hp)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Reference natural linewidth
    ax.axhline(6.065, color='darkgreen', ls='--', lw=2.2, alpha=0.9,
               label='Natural Linewidth $\\Gamma_{\\mathrm{nat}} = 6.07$ MHz')

    # 87Rb data and fits
    ax.plot(powers, f87, 'o', color='crimson', ms=7.5, mec='black', mew=1.0, alpha=0.9,
            label='$^{87}$Rb ($F=2 \\rightarrow F\'=3$) Measured Data')
    label_fit87_all = (f'$^{{87}}$Rb All Points: '
                       f'$\\Gamma_0 = {popt87_all[0]:.2f} \\pm {perr87_all[0]:.2f}$ MHz, '
                       f'$P_{{\\mathrm{{sat}}}} = {popt87_all[1]:.0f}$ $\\mu$W')
    ax.plot(p_dense, fit87_all_dense, '-', color='crimson', lw=2.2, label=label_fit87_all)

    label_fit87_hp = (f'$^{{87}}$Rb High Power ($P \\geq 200\\mu$W): '
                      f'$\\Gamma_0 = {popt87_hp[0]:.2f} \\pm {perr87_hp[0]:.2f}$ MHz, '
                      f'$P_{{\\mathrm{{sat}}}} = {popt87_hp[1]:.0f}$ $\\mu$W')
    ax.plot(p_dense, fit87_hp_dense, ':', color='crimson', lw=2.4, label=label_fit87_hp)

    # 85Rb data and fits
    ax.plot(powers[mask85_all], f85[mask85_all], 's', color='tab:blue', ms=7.0, mec='black', mew=1.0, alpha=0.9,
            label='$^{85}$Rb ($F=3 \\rightarrow F\'=4$) Measured Data')
    if np.any(~mask85_all):
        ax.plot(powers[~mask85_all], f85[~mask85_all], 's', color='lightskyblue', ms=6.5, mec='gray', mew=0.8,
                label='$^{85}$Rb Low-Power Data (SNR < 3, Excluded)')
    label_fit85_all = (f'$^{{85}}$Rb All Points ($P \\geq 35\\mu$W): '
                       f'$\\Gamma_0 = {popt85_all[0]:.2f} \\pm {perr85_all[0]:.2f}$ MHz, '
                       f'$P_{{\\mathrm{{sat}}}} = {popt85_all[1]:.0f}$ $\\mu$W')
    ax.plot(p_dense, fit85_all_dense, '-', color='tab:blue', lw=2.2, label=label_fit85_all)

    label_fit85_hp = (f'$^{{85}}$Rb High Power ($P \\geq 200\\mu$W): '
                      f'$\\Gamma_0 = {popt85_hp[0]:.2f} \\pm {perr85_hp[0]:.2f}$ MHz, '
                      f'$P_{{\\mathrm{{sat}}}} = {popt85_hp[1]:.0f}$ $\\mu$W')
    ax.plot(p_dense, fit85_hp_dense, ':', color='tab:blue', lw=2.4, label=label_fit85_hp)

    ax.set_title("Cycling Transition Linewidth (FWHM) vs. Laser Beam Power\nComparing All-Points Fit vs. High-Power Extrapolation ($P \\geq 200\\mu$W)",
                 fontsize=12.5, fontweight='bold')
    ax.set_xlabel("Laser Beam Power ($\\mu$W)", fontsize=11.5)
    ax.set_ylabel("Measured FWHM (MHz)", fontsize=11.5)
    ax.set_xlim(-20, np.max(powers) * 1.05)
    ax.set_ylim(4.0, 9.8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)

    # Annotate zero-power extrapolations
    ax.annotate(f"All Points: $\\Gamma_0 = {popt87_all[0]:.2f}$ MHz\nHigh Power: $\\Gamma_0 = {popt87_hp[0]:.2f}$ MHz",
                xy=(0, popt87_hp[0]), xytext=(70, popt87_hp[0] + 0.30),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5),
                fontsize=9.5, fontweight='bold', color='crimson',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='crimson', alpha=0.9))

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

    # Save summary CSV
    df_out = pd.DataFrame([{
        'run': r['run'],
        'power_uW': r['power_uW'],
        'fwhm_87_MHz': r['fwhm_87'],
        'amp_87_mV': r['amp_87'],
        'fwhm_85_MHz': r['fwhm_85'],
        'amp_85_mV': r['amp_85']
    } for r in records])
    csv_path = 'figures/fwhm_noise_filtered_summary.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"Saved summary metrics to: {csv_path}")

    args = sys.argv[1:]
    if '--save' in args:
        print("Saving all 22 interactive run figures...")
        for rec in records:
            p_save = os.path.join(out_dir_runs, f"fwhm_steps_run_{rec['run']:02d}_{rec['power_uW']:06.2f}uW.png")
            plot_single_run_acquisition(rec, show=False, save_path=p_save)
        plot_fwhm_vs_power(records, show=False)
        print("All figures saved.")
        return

    if '--summary' in args:
        print("Displaying summary FWHM vs Power curve...")
        plot_fwhm_vs_power(records, show=True)
        return

    # Check if a specific run number was requested (e.g. `python ... 10`)
    run_num_arg = None
    for a in args:
        try:
            val = int(a)
            if 1 <= val <= 22:
                run_num_arg = val
                break
        except ValueError:
            pass

    if run_num_arg is not None:
        rec = next(r for r in records if r['run'] == run_num_arg)
        print(f"Displaying interactive FWHM acquisition steps for Run {run_num_arg} ({rec['power_uW']:.2f} uW)...")
        plot_single_run_acquisition(rec, show=True)
        return

    # Default action: Display all 22 power sets in order (sorted by power), followed by summary plot
    sorted_records = sorted(records, key=lambda x: x['power_uW'])
    total = len(sorted_records)
    print(f"\n[Interactive Mode: Displaying all {total} power sets in order of increasing power]")
    print("Close each figure window to advance to the next power run, or press Ctrl+C in terminal to stop.")
    for idx, rec in enumerate(sorted_records, 1):
        print(f"[{idx:02d}/{total}] Run {rec['run']:02d} (Power: {rec['power_uW']:6.2f} uW) — "
              f"87Rb: {rec['fwhm_87']:.2f} MHz, 85Rb: {rec['fwhm_85']:.2f} MHz")
        plot_single_run_acquisition(rec, show=True)

    print("\nDisplaying overall FWHM vs. Power summary curve with all-points and high-power extrapolations...")
    plot_fwhm_vs_power(records, show=True)

    sample_rec = next(r for r in records if r['run'] == 10)
    print("2. Displaying step-by-step acquisition for low-power Run 10 (19.57 uW)...")
    plot_single_run_acquisition(sample_rec, show=True)


if __name__ == '__main__':
    main()
