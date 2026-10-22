import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import physics_model as pm


def load_dataset():
    """Loads oscilloscope traces and extracts the calibrated frequency axis."""
    df_fp1 = pd.read_csv("data/FP1.CSV")
    fabry = df_fp1["C2 in V"].values
    df_fp2 = pd.read_csv("data/FP02.CSV")
    c3_pumped = df_fp2["C3 in V"].values
    df_fp3 = pd.read_csv("data/FP03.CSV")
    c3_unpumped = df_fp3["C3 in V"].values

    # Find FP transmission fringes
    peaks, _ = find_peaks(fabry, prominence=1e-2)
    index = np.arange(len(fabry))
    peak_index = np.arange(len(peaks))
    linear_freq = np.poly1d(np.polyfit(peaks, peak_index, 1))

    # Detect mode-hop edges
    edge_indices = find_peaks(peak_index - linear_freq(peaks))[0]
    edge_indices[0] += 1
    edge_indices[1] -= 1
    left_edge, right_edge = peaks[edge_indices]

    # Slice out single-mode usable region
    peaks_cut = peaks[edge_indices[0] : edge_indices[1] + 1]
    peak_index_cut = peak_index[edge_indices[0] : edge_indices[1] + 1]
    index_cut = index[left_edge : right_edge + 1]
    diff_cut = c3_pumped[left_edge : right_edge + 1] - c3_unpumped[left_edge : right_edge + 1]

    # Polynomial frequency calibration (mapping scope sample index -> FP fringe index)
    freq_fit = np.poly1d(np.polyfit(peaks_cut, peak_index_cut, 5))
    freq_fringes = freq_fit(index_cut)

    return freq_fringes, diff_cut, c3_pumped[left_edge : right_edge + 1], c3_unpumped[left_edge : right_edge + 1]


def run_global_fit(B_gauss):
    """
    Fits a unified forward physical model across all four rubidium D2 manifolds simultaneously,
    sharing the Fabry-Pérot FSR, independent polarization ellipticities (pump & probe),
    homogeneous linewidth gamma_eff, and transit relaxation rate gamma_tr.
    """
    freq_fringes, diff_cut, _, _ = load_dataset()

    print("=" * 80)
    print("GLOBAL MULTI-MANIFOLD FORWARD MODEL FIT (All 4 Manifolds Simultaneously)")
    print(f"Applied Longitudinal Magnetic Field: B = {B_gauss:.1f} G")
    print("=" * 80)

    # Initial parameter guesses (explicitly defined at top level from polarimeter and physics)
    # [fsr, gamma_eff, gamma_tr, s_pump, kappa_probe, epsilon_pump, epsilon_probe, delta_theta, f0_0, f0_1, f0_2, f0_3]
    eps_pump_init = np.radians(-30.6)      # Polarimeter measurement for pump (~6.6°)
    eps_probe_init = np.radians(-14.9)    # Polarimeter measurement for probe (~14.9°)
    kappa_probe_init = 0.33              # Polarimeter measured probe/pump power ratio (110 uW / 330 uW)
    delta_theta_init = np.radians(50.0) # Polarimeter measured major-axis angle difference (~100°)

    p0 = [
        375.0,                         # FSR (MHz)
        12.0,                          # gamma_eff (MHz)
        0.15,                          # gamma_tr (MHz)
        2.0,                           # s_pump
        kappa_probe_init,              # kappa_probe
        eps_pump_init,                 # epsilon_pump (rad)
        eps_probe_init,                # epsilon_probe (rad)
        delta_theta_init,              # delta_theta (rad)
        pm.MANIFOLDS[0].f0_guess,      # f0: 87Rb F=2
        pm.MANIFOLDS[1].f0_guess,      # f0: 85Rb F=3
        pm.MANIFOLDS[2].f0_guess,      # f0: 85Rb F=2
        pm.MANIFOLDS[3].f0_guess,      # f0: 87Rb F=1
    ]

    bounds_lower = [
        360.0, 5.0, 0.01, 0.1, 0.05, -np.pi / 4.0, -np.pi / 4.0, 0.0,
        9.5, 12.5, 20.0, 26.5,
    ]
    bounds_upper = [
        400.0, 30.0, 2.0, 20.0, 0.8, np.pi / 4.0, np.pi / 4.0, np.pi,
        11.0, 14.0, 21.5, 28.0,
    ]

    subsample_step = 50
    max_nfev = 60

    fit_res = pm.fit_global_forward_model(
        freq_fringes,
        diff_cut,
        p0,
        bounds_lower,
        bounds_upper,
        B_gauss,
        subsample_step,
        max_nfev,
    )

    print("\n" + "=" * 80)
    print("GLOBAL CONVERGED PARAMETERS (SHARED ACROSS ALL 4 MANIFOLDS):")
    print(f"  Fabry-Perot FSR          : {fit_res.fsr:.3f} MHz (nominal ~374.74 MHz)")
    print(f"  Pump Ellipticity         : {fit_res.epsilon_pump:.4f} rad ({np.degrees(fit_res.epsilon_pump):.2f}°)")
    print(f"  Probe Ellipticity        : {fit_res.epsilon_probe:.4f} rad ({np.degrees(fit_res.epsilon_probe):.2f}°)")
    print(f"  Probe / Pump Ratio kappa : {fit_res.kappa_probe:.3f} (measured power ratio ~0.33)")
    print(f"  Relative Orientation     : {fit_res.delta_theta:.4f} rad ({np.degrees(fit_res.delta_theta):.2f}°)")
    print(f"  Homogeneous Linewidth    : {fit_res.gamma_eff:.2f} MHz (natural Gamma_0 = 6.07 MHz)")
    print(f"  Beam Transit Rate        : {fit_res.gamma_tr:.4f} MHz")
    print(f"  Global Pump Saturation   : {fit_res.s_pump:.2f} (I_pump / I_sat,0)")
    print(f"  Fit Optimization Success : {fit_res.success} (cost = {fit_res.cost:.4e}, nfev = {fit_res.nfev})")
    print("-" * 80)
    print("LOCAL MANIFOLD-SPECIFIC PARAMETERS:")
    for i, m in enumerate(pm.MANIFOLDS):
        print(f"  [{m.name}]")
        print(f"    Line Center (f0)     : {fit_res.f0[i]:.4f} fringes")
        print(f"    Amplitude Scale      : {fit_res.amp[i]:.2f} V")
        print(f"    Baseline Offset      : {fit_res.bg[i]:.5f} V")
    print("=" * 80 + "\n")

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for i, m in enumerate(pm.MANIFOLDS):
        r_min, r_max = m.fringe_window
        mask = (freq_fringes >= r_min) & (freq_fringes <= r_max)
        x_m = freq_fringes[mask]
        y_m = diff_cut[mask]

        # Evaluate model with converged global and local parameters on dense grid
        x_eval = np.linspace(r_min, r_max, 350)
        nu_eval = (x_eval - fit_res.f0[i]) * fit_res.fsr
        _, _, diff_sim = pm.simulate_manifold_zeeman(
            nu_eval,
            m,
            B_gauss,
            fit_res.epsilon_pump,
            fit_res.epsilon_probe,
            fit_res.delta_theta,
            295.0,
            fit_res.s_pump,
            fit_res.kappa_probe,
            fit_res.gamma_eff,
            fit_res.gamma_tr,
            151,
        )
        y_fit = fit_res.amp[i] * diff_sim + fit_res.bg[i]

        ax = axs.flat[i]
        ax.plot(x_m, y_m, label="Measured Diff", color="navy", alpha=0.85)
        ax.plot(x_eval, y_fit, "--", label="Global Model Fit", color="crimson", linewidth=2.0)
        ax.set_title(
            f"{m.name} ($f_0={fit_res.f0[i]:.3f}$, $A={fit_res.amp[i]:.1f}$ V)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Fabry-Pérot Fringe Index")
        ax.set_ylabel("Photodiode Voltage (V)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    plt.suptitle(
        f"Global Fit: FSR = {fit_res.fsr:.1f} MHz, $s_{{pump}} = {fit_res.s_pump:.2f}$, $\\kappa = {fit_res.kappa_probe:.2f}$, $\\epsilon_{{pump}} = {np.degrees(fit_res.epsilon_pump):.1f}^\\circ$, $\\epsilon_{{probe}} = {np.degrees(fit_res.epsilon_probe):.1f}^\\circ$, $\\Delta\\theta = {np.degrees(fit_res.delta_theta):.1f}^\\circ$, $\\Gamma_{{eff}} = {fit_res.gamma_eff:.1f}$ MHz",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("global_forward_model_fit.png", dpi=200)
    print("Saved global multi-manifold comparison plot to 'global_forward_model_fit.png'.", flush=True)
    plt.close(fig)

    return fit_res


if __name__ == "__main__":
    # Fit global unified forward model across all 4 manifolds simultaneously
    run_global_fit(0.0)
