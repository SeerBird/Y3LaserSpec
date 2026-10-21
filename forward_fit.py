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


def run_fits(B_gauss=0.0, epsilon_pol=0.0):
    """
    Fits the Zeeman-resolved rate equation model across all 4 manifolds with B=0 (or custom B).
    """
    freq_fringes, diff_cut, c3_pumped, c3_unpumped = load_dataset()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    print("=" * 75)
    print(f"ZEEMAN-RESOLVED FORWARD MODEL FIT (B = {B_gauss:.1f} G, epsilon = {np.degrees(epsilon_pol):.1f}°)")
    print("=" * 75)

    for i, m in enumerate(pm.MANIFOLDS):
        r_min, r_max = m.fringe_window
        mask = (freq_fringes >= r_min) & (freq_fringes <= r_max)
        x_m = freq_fringes[mask]
        y_m = diff_cut[mask]

        # Fit Zeeman-resolved model with finite probe ratio (10% / 45% = 2/9)
        kappa_probe = 10.0 / 45.0
        res = pm.fit_manifold_zeeman(
            x_m, y_m, manifold_index=i, B_gauss=B_gauss, epsilon_pol=epsilon_pol, kappa_probe=kappa_probe
        )
        fsr, f0, s_pump, gamma_eff, gamma_tr, amp, bg = res.x

        # Dense curve for visualization
        x_eval = np.linspace(r_min, r_max, 350)
        nu_eval = (x_eval - f0) * fsr
        _, _, diff_sim = pm.simulate_manifold_zeeman(
            nu_eval,
            m,
            B_gauss=B_gauss,
            epsilon_pol=epsilon_pol,
            T=295.0,
            s_pump=s_pump,
            kappa_probe=kappa_probe,
            gamma_eff=gamma_eff,
            gamma_tr=gamma_tr,
            N_v=151,
        )
        y_fit = amp * diff_sim + bg

        print(f"Manifold: {m.name}")
        print(f"  Etalon FSR          : {fsr:.2f} MHz (nominal ~374.74 MHz)")
        print(f"  Line Center (f0)    : {f0:.4f} fringes")
        print(f"  Saturation (s_pump) : {s_pump:.2f} (I_pump / I_sat)")
        print(f"  Effective Linewidth : {gamma_eff:.2f} MHz")
        print(f"  Transit Decay Rate  : {gamma_tr:.3f} MHz")
        print(f"  Residual Norm (cost): {res.cost:.4e}")
        print("-" * 75)

        # Plot comparison
        ax = axs.flat[i]
        ax.plot(x_m, y_m, label="Measured Diff", color="navy", alpha=0.85)
        ax.plot(x_eval, y_fit, "--", label="Zeeman Model Fit", color="crimson", linewidth=2.0)
        ax.set_title(
            f"{m.name} (FSR = {fsr:.1f} MHz, $s_{{pump}} = {s_pump:.2f}$, $\\Gamma_{{eff}} = {gamma_eff:.1f}$ MHz)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Fabry-Pérot Fringe Index")
        ax.set_ylabel("Photodiode Voltage (V)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    plt.suptitle(
        f"Doppler-Free Saturated Absorption: Zeeman-Resolved Rate Model (B = {B_gauss:.1f} G, Linear Pol)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("forward_model_fit.png", dpi=200)
    print("\nSaved high-resolution comparison plot to 'forward_model_fit.png'.", flush=True)
    plt.close(fig)


def run_global_fit(B_gauss=0.0):
    """
    Fits a unified forward physical model across all four rubidium D2 manifolds simultaneously,
    sharing the Fabry-Pérot FSR, polarization ellipticity epsilon_pol, homogeneous linewidth
    gamma_eff, and transit relaxation rate gamma_tr.
    """
    freq_fringes, diff_cut, c3_pumped, c3_unpumped = load_dataset()

    print("=" * 80)
    print("GLOBAL MULTI-MANIFOLD FORWARD MODEL FIT (All 4 Manifolds Simultaneously)")
    print(f"Applied Longitudinal Magnetic Field: B = {B_gauss:.1f} G")
    print("=" * 80)

    fit_res = pm.fit_global_forward_model(
        freq_fringes,
        diff_cut,
        B_gauss=B_gauss,
        kappa_probe=10.0 / 45.0,
    )

    print("\n" + "=" * 80)
    print("GLOBAL CONVERGED PARAMETERS (SHARED ACROSS ALL 4 MANIFOLDS):")
    print(f"  Fabry-Perot FSR          : {fit_res.fsr:.3f} MHz (nominal ~374.74 MHz)")
    print(f"  Polarization Ellipticity : {fit_res.epsilon_pol:.4f} rad ({np.degrees(fit_res.epsilon_pol):.2f}°)")
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
            B_gauss=B_gauss,
            epsilon_pol=fit_res.epsilon_pol,
            T=295.0,
            s_pump=fit_res.s_pump,
            kappa_probe=10.0 / 45.0,
            gamma_eff=fit_res.gamma_eff,
            gamma_tr=fit_res.gamma_tr,
            N_v=151,
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
        f"Global Forward Model Fit (Shared FSR = {fit_res.fsr:.1f} MHz, $s_{{pump}} = {fit_res.s_pump:.2f}$, $\\epsilon_{{pol}} = {np.degrees(fit_res.epsilon_pol):.1f}^\\circ$, $\\Gamma_{{eff}} = {fit_res.gamma_eff:.1f}$ MHz, $\\gamma_{{tr}} = {fit_res.gamma_tr:.3f}$ MHz)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("global_forward_model_fit.png", dpi=200)
    print("Saved global multi-manifold comparison plot to 'global_forward_model_fit.png'.", flush=True)
    plt.close(fig)

    return fit_res


def simulate_magnetic_field_effect():
    """
    Demonstrates the predicted effect of turning on the Helmholtz coils (B > 0).
    Saves a comparison plot for 87Rb F=2 showing Zeeman splitting.
    """
    m = pm.MANIFOLDS[0]  # 87Rb F=2
    nu = np.linspace(-550.0, 150.0, 400)

    fig, ax = plt.subplots(figsize=(10, 5))
    fields = [0.0, 10.0, 30.0]
    colors = ['navy', 'forestgreen', 'crimson']

    for B, col in zip(fields, colors):
        _, _, diff = pm.simulate_manifold_zeeman(
            nu, m, B_gauss=B, epsilon_pol=0.0, T=295.0, s_pump=1.5, gamma_eff=15.0, gamma_tr=0.1, N_v=51
        )
        ax.plot(nu, diff, label=f"B = {B:.0f} Gauss", color=col, linewidth=1.8)

    ax.set_title("Predicted Zeeman Splitting in 87Rb F=2 with Longitudinal B-Field (k || B)")
    ax.set_xlabel("Relative Frequency (MHz)")
    ax.set_ylabel("Saturated Absorption Signal (arb)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig("zeeman_splitting_prediction.png", dpi=200)
    print("Saved magnetic field prediction plot to 'zeeman_splitting_prediction.png'.", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    # 1. Fit global unified forward model across all 4 manifolds simultaneously
    run_global_fit(B_gauss=0.0)

    # 2. Simulate what happens when coils are turned on (B = 0, 10, 30 G)
    simulate_magnetic_field_effect()
