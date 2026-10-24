import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import physics_model as pm


def load_dataset(
    fp_csv: str = "data/FP1.CSV",
    pumped_csv: str = "data/FP02.CSV",
    unpumped_csv: str = "data/FP03.CSV",
    fabry_channel: str = "C2 in V",
    signal_channel: str = "C3 in V",
):
    """
    Loads oscilloscope traces from the specified CSV files and extracts the
    calibrated Fabry-Perot frequency axis and differential absorption signal.
    """
    df_fp1 = pd.read_csv(fp_csv)
    fabry = df_fp1[fabry_channel].values
    df_fp2 = pd.read_csv(pumped_csv)
    c3_pumped = df_fp2[signal_channel].values
    df_fp3 = pd.read_csv(unpumped_csv)
    c3_unpumped = df_fp3[signal_channel].values

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


def run_global_fit(
    freq_fringes: np.ndarray,
    diff_cut: np.ndarray,
    fringe_windows: list[tuple[float, float]] | np.ndarray,
    p0: list[float],
    bounds_lower: list[float],
    bounds_upper: list[float],
    fixed_params: dict[str, float] | None = None,
    plot_filename: str = os.path.join("figures", "global_forward_model_fit.png"),
    subsample_step: int = 50,
    max_nfev: int = 60,
    model_type: str = "coherent_operator",
) -> pm.GlobalFitResult:
    """
    Fits a unified forward physical model across all four rubidium D2 manifolds simultaneously,
    sharing the Fabry-Pérot FSR, independent polarization ellipticities (pump & probe),
    magnetic field B (or fixed), homogeneous linewidth gamma_eff, and transit relaxation rate gamma_tr.
    
    Supports model_type="coherent_operator" (Approach 3) and model_type="density_matrix" (Approach 2).
    """
    fixed_dict = fixed_params if fixed_params is not None else {}
    if not os.path.dirname(plot_filename):
        plot_filename = os.path.join("figures", plot_filename)
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    print("=" * 80)
    print(f"GLOBAL MULTI-MANIFOLD FORWARD MODEL FIT (Model: {model_type})")
    if fixed_dict:
        print(f"Fixed Parameters: {fixed_dict}")
    print(f"Output Plot Filename: {plot_filename}")
    print("=" * 80)

    fit_res = pm.fit_global_forward_model(
        freq_fringes,
        diff_cut,
        fringe_windows,
        p0,
        bounds_lower,
        bounds_upper,
        fixed_dict,
        subsample_step,
        max_nfev,
        model_type,
    )

    print("\n" + "=" * 80)
    print("GLOBAL CONVERGED PARAMETERS (SHARED ACROSS ALL 4 MANIFOLDS):")
    print(f"  Physical Model Type      : {fit_res.model_type}")
    print(f"  Fabry-Perot FSR          : {fit_res.fsr:.3f} MHz (nominal ~374.74 MHz)")
    print(f"  Magnetic Field Vector B  : ({fit_res.Bx_gauss:.2f}, {fit_res.By_gauss:.2f}, {fit_res.Bz_gauss:.2f}) G (|B| = {fit_res.B_gauss:.2f} G)")
    print(f"  Pump Ellipticity         : {fit_res.epsilon_pump:.4f} rad ({np.degrees(fit_res.epsilon_pump):.2f}°)")
    print(f"  Probe Ellipticity        : {fit_res.epsilon_probe:.4f} rad ({np.degrees(fit_res.epsilon_probe):.2f}°)")
    print(f"  Probe / Pump Ratio kappa : {fit_res.kappa_probe:.3f} (measured power ratio ~0.33)")
    print(f"  Homogeneous Linewidth    : {fit_res.gamma_eff:.2f} MHz (natural Gamma_0 = 6.07 MHz)")
    print(f"  Beam Transit Rate        : {fit_res.gamma_tr:.4f} MHz")
    print(f"  Global Pump Saturation   : {fit_res.s_pump:.2f} (I_pump / I_sat,0)")
    print(f"  Fit Optimization Success : {fit_res.success} (cost = {fit_res.cost:.4e}, nfev = {fit_res.nfev})")
    if fit_res.fixed_params:
        print(f"  Fixed Parameters Record  : {fit_res.fixed_params}")
    print("-" * 80)
    print("LOCAL MANIFOLD-SPECIFIC PARAMETERS:")
    for i, m in enumerate(pm.MANIFOLDS):
        print(f"  [{m.name}]")
        print(f"    Line Center (f0)     : {fit_res.f0[i]:.4f} fringes")
        print(f"    Amplitude Scale      : {fit_res.amp[i]:.2f} V")
        print(f"    Baseline Offset      : {fit_res.bg[i]:.5f} V")
    print("=" * 80 + "\n")

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for i, (m, (r_min, r_max)) in enumerate(zip(pm.MANIFOLDS, fringe_windows)):
        mask = (freq_fringes >= r_min) & (freq_fringes <= r_max)
        x_m = freq_fringes[mask]
        y_m = diff_cut[mask]

        # Evaluate model with converged global and local parameters on dense grid
        x_eval = np.linspace(r_min, r_max, 350)
        nu_eval = (x_eval - fit_res.f0[i]) * fit_res.fsr
        _, _, diff_sim = pm.simulate_manifold_zeeman(
            nu_eval,
            m,
            fit_res.Bx_gauss,
            fit_res.By_gauss,
            fit_res.Bz_gauss,
            fit_res.epsilon_pump,
            fit_res.epsilon_probe,
            295.0,
            fit_res.s_pump,
            fit_res.kappa_probe,
            fit_res.gamma_eff,
            fit_res.gamma_tr,
            151,
            fit_res.model_type,
        )
        y_fit = fit_res.amp[i] * diff_sim + fit_res.bg[i]

        ax = axs.flat[i]
        ax.plot(x_m, y_m, label="Measured Diff", color="navy", alpha=0.85)
        ax.plot(x_eval, y_fit, "--", label=f"Model Fit ({fit_res.model_type})", color="crimson", linewidth=2.0)
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
        f"Global Fit ({fit_res.model_type}): FSR = {fit_res.fsr:.1f} MHz, $\\vec{{B}} = ({fit_res.Bx_gauss:.1f}, {fit_res.By_gauss:.1f}, {fit_res.Bz_gauss:.1f})$ G ($|B|={fit_res.B_gauss:.1f}$ G), $s_{{pump}} = {fit_res.s_pump:.2f}$, $\\kappa = {fit_res.kappa_probe:.2f}$, $\\epsilon_{{pump}} = {np.degrees(fit_res.epsilon_pump):.1f}^\\circ$, $\\epsilon_{{probe}} = {np.degrees(fit_res.epsilon_probe):.1f}^\\circ$, $\\Gamma_{{eff}} = {fit_res.gamma_eff:.1f}$ MHz",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=200)
    print(f"Saved global multi-manifold comparison plot to '{plot_filename}'.", flush=True)
    plt.close(fig)

    return fit_res


def run_zero_field(model_type: str = "coherent_operator"):
    """
    Runs the global multi-manifold forward model fit on the standard zero-field dataset (FP1, FP02, FP03)
    with magnetic field B fixed to 0.0 G via run_global_fit arguments.
    """
    freq_fringes_std, diff_cut_std, _, _ = load_dataset(
        fp_csv="data/FP1.CSV",
        pumped_csv="data/FP02.CSV",
        unpumped_csv="data/FP03.CSV",
    )

    fringe_windows_std = [
        (9.0, 10.6),   # 87Rb F=2
        (12.6, 13.6),  # 85Rb F=3
        (20.6, 21.4),  # 85Rb F=2
        (27.1, 28.1),  # 87Rb F=1
    ]

    # Initial parameter guesses (explicitly defined at top level from polarimeter and physics)
    # [fsr, gamma_eff, gamma_tr, s_pump, kappa_probe, epsilon_pump, epsilon_probe, Bx_gauss, By_gauss, Bz_gauss, f0_0, f0_1, f0_2, f0_3]
    eps_pump_init = np.radians(6.6)      # Polarimeter measurement for pump (~6.6°)
    eps_probe_init = np.radians(14.9)    # Polarimeter measurement for probe (~14.9°)
    kappa_probe_init = 0.33              # Polarimeter measured probe/pump power ratio (110 uW / 330 uW)

    p0_std = [
        375.0,              # FSR (MHz)
        12.0,               # gamma_eff (MHz)
        0.15,               # gamma_tr (MHz)
        2.0,                # s_pump
        kappa_probe_init,   # kappa_probe
        eps_pump_init,      # epsilon_pump (rad)
        eps_probe_init,     # epsilon_probe (rad)
        0.0,                # Bx_gauss (G) - free, initialized at 0.0 G
        0.0,                # By_gauss (G) - free, initialized at 0.0 G
        0.0,                # Bz_gauss (G) - free, initialized at 0.0 G
        10.33,              # f0: 87Rb F=2
        13.34,              # f0: 85Rb F=3
        20.88,              # f0: 85Rb F=2
        27.28,              # f0: 87Rb F=1
    ]

    bounds_lower_std = [
        360.0, 5.0, 0.01, 0.1, 0.05, -np.pi / 4.0, -np.pi / 4.0,
        -3.0, -3.0, -3.0,
        9.5, 12.5, 20.0, 26.5,
    ]
    bounds_upper_std = [
        400.0, 30.0, 2.0, 20.0, 0.8, np.pi / 4.0, np.pi / 4.0,
        3.0, 3.0, 3.0,
        11.0, 14.0, 21.5, 28.0,
    ]

    return run_global_fit(
        freq_fringes=freq_fringes_std,
        diff_cut=diff_cut_std,
        fringe_windows=fringe_windows_std,
        p0=p0_std,
        bounds_lower=bounds_lower_std,
        bounds_upper=bounds_upper_std,
        fixed_params={},
        plot_filename=os.path.join("figures", f"zero_field_{model_type}_fit.png"),
        subsample_step=50,
        max_nfev=60,
        model_type=model_type,
    )


def run_mag1(model_type: str = "coherent_operator"):
    """
    Runs the global multi-manifold forward model fit on the Session 8 magnetic field dataset (FPMAG1, PMAG1, UMAG1)
    with magnetic field B as a free fitted parameter.
    """
    freq_fringes_mag1, diff_cut_mag1, _, _ = load_dataset(
        fp_csv="data/Session 8/FPMAG1.CSV",
        pumped_csv="data/Session 8/PMAG1.CSV",
        unpumped_csv="data/Session 8/UMAG1.CSV",
    )

    fringe_windows_mag1 = [
        (5.5, 7.3),    # 87Rb F=2
        (9.0, 10.5),   # 85Rb F=3
        (16.7, 18.5),  # 85Rb F=2
        (23.1, 25.2),  # 87Rb F=1
    ]

    eps_pump_init = np.radians(6.6)
    eps_probe_init = np.radians(14.9)
    kappa_probe_init = 0.33

    p0_mag1 = [
        375.0,              # FSR (MHz)
        12.0,               # gamma_eff (MHz)
        0.15,               # gamma_tr (MHz)
        2.0,                # s_pump
        kappa_probe_init,   # kappa_probe
        eps_pump_init,      # epsilon_pump (rad)
        eps_probe_init,     # epsilon_probe (rad)
        0.5,                # Bx_gauss (G) - free fitted parameter, initialized at ~0.5 G
        0.0,                # By_gauss (G) - free fitted parameter, initialized at 0.0 G
        2.5,                # Bz_gauss (G) - free fitted parameter, initialized at ~2.5 G
        6.92,               # f0: 87Rb F=2
        9.92,               # f0: 85Rb F=3
        17.49,              # f0: 85Rb F=2
        23.91,              # f0: 87Rb F=1
    ]

    bounds_lower_mag1 = [
        360.0, 5.0, 0.01, 0.1, 0.05, -np.pi / 4.0, -np.pi / 4.0,
        -5.0, -5.0, -50.0,
        5.5, 9.0, 16.7, 23.1,
    ]
    bounds_upper_mag1 = [
        400.0, 30.0, 2.0, 20.0, 0.8, np.pi / 4.0, np.pi / 4.0,
        5.0, 5.0, 50.0,
        7.3, 10.5, 18.5, 25.2,
    ]

    return run_global_fit(
        freq_fringes=freq_fringes_mag1,
        diff_cut=diff_cut_mag1,
        fringe_windows=fringe_windows_mag1,
        p0=p0_mag1,
        bounds_lower=bounds_lower_mag1,
        bounds_upper=bounds_upper_mag1,
        fixed_params={},
        plot_filename=os.path.join("figures", f"mag1_{model_type}_fit.png"),
        subsample_step=50,
        max_nfev=60,
        model_type=model_type,
    )


if __name__ == "__main__":
    run_zero_field(model_type="coherent_operator")
    run_mag1(model_type="coherent_operator")
