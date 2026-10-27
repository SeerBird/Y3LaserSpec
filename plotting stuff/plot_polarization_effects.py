import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import physics_model as pm
import forward_fit as ff


def plot_polarization_effects(resolution:int):
    """
    Simulates and visualizes how pump and probe beam polarization states (controlled in lab
    via quarter-wave plates) affect the Doppler-free peak shapes in the lower ground state
    manifolds: 87Rb F=1 and 85Rb F=2.
    """
    # Load experimental data
    freq_fringes, diff_cut, _, _ = ff.load_dataset()

    # Representative physical parameters from global fit
    B_gauss = 0.0
    T = 295.0
    s_pump = 2.0
    kappa_probe = 0.30
    gamma_eff = 12.0
    gamma_tr = 0.15
    N_v = 151
    fsr = 375.0

    # Key polarization scenarios: (label, eps_pump_deg, eps_probe_deg, color, linestyle)
    polarization_scenarios = [
        ("Linear || Linear (eps=0°, 0°)", 0.0, 0.0, "black", "-"),
        ("Circular sigma+ / sigma+ (+45°, +45°)", 45.0, 45.0, "crimson", "--"),
        ("Circular sigma+ / sigma- (+45°, -45°)", 45.0, -45.0, "forestgreen", "-."),
        ("Lab Polarimeter (+6.6°, +14.9°)", 6.6, 14.9, "royalblue", "-"),
        ("Opposite Helicity (+6.6°, -14.9°)", 6.6, -14.9, "darkorange", ":"),
    ]

    # Target lower ground state manifolds: (manifold, fringe_window, f0_guess, label)
    target_manifolds = [
        (pm.MANIFOLDS[3], (27.1, 28.1), 27.28, "87Rb F=1 (Ground F=1 -> Excited F'=0, 1, 2)"),
        (pm.MANIFOLDS[2], (20.6, 21.4), 20.88, "85Rb F=2 (Ground F=2 -> Excited F'=1, 2, 3)"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))

    for ax, (m, (r_min, r_max), f0, title_label) in zip(axes, target_manifolds):
        # Extract experimental slice
        mask = (freq_fringes >= r_min) & (freq_fringes <= r_max)

        x_exp = freq_fringes[mask]
        y_exp = diff_cut[mask]
        step = max(1, len(x_exp)//resolution)
        x_exp = x_exp[::step]
        y_exp = y_exp[::step]

        # Frequency evaluation grid around manifold line center
        nu_eval = (x_exp - f0) * fsr

        # Baseline & amplitude normalization from linear fit for lab polarimeter case
        eps_pump_ref = np.radians(6.6)
        eps_probe_ref = np.radians(14.9)
        delta_theta_ref = np.radians(100.0)
        _, _, diff_ref = pm.simulate_manifold_zeeman(
            nu_eval,
            m,
            B_gauss,
            eps_pump_ref,
            eps_probe_ref,
            delta_theta_ref,
            T,
            s_pump,
            kappa_probe,
            gamma_eff,
            gamma_tr,
            N_v,
        )
        X = np.column_stack([diff_ref, np.ones_like(diff_ref)])
        coeffs, _, _, _ = np.linalg.lstsq(X, y_exp, rcond=None)
        amp_scale, bg_offset = coeffs[0], coeffs[1]

        # Plot experimental data
        ax.plot(
            nu_eval,
            y_exp,
            label="Experimental Data (Scope Trace)",
            color="lightsteelblue",
            linewidth=2.5,
            alpha=0.9,
        )

        # Simulate and overlay each polarization combination
        for label, ep_deg, epr_deg, col, ls in polarization_scenarios:
            ep_rad = np.radians(ep_deg)
            epr_rad = np.radians(epr_deg)
            dth = 0.0 if "Linear || Linear" in label else delta_theta_ref
            _, _, diff_sim = pm.simulate_manifold_zeeman(
                nu_eval,
                m,
                B_gauss,
                ep_rad,
                epr_rad,
                dth,
                T,
                s_pump,
                kappa_probe,
                gamma_eff,
                gamma_tr,
                N_v,
            )
            # Scale simulated diff using same physical amplitude & baseline
            y_sim = amp_scale * diff_sim + bg_offset
            ax.plot(nu_eval, y_sim, label=label, color=col, linestyle=ls, linewidth=1.8)

        ax.set_title(f"{title_label} - Sensitivity to QWP Polarizations", fontsize=12, fontweight="bold")
        ax.set_xlabel("Relative Detuning from Reference Transition (MHz)", fontsize=11)
        ax.set_ylabel("Photodiode Voltage (V)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)

    plt.suptitle(
        "Impact of Quarter-Wave Plate Polarizations on Lower Ground-State Hyperfine Manifolds\n"
        "(Dark States & Optical Pumping Dramatically Alter Crossover Peak Heights and Inversions)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    output_path = os.path.join("../figures", "polarization_effects_lower_states.png")
    plt.savefig(output_path, dpi=200)
    print(f"Saved discrete polarization comparison plot to '{output_path}'.", flush=True)
    plt.close(fig)


def plot_qwp_rotation_sweep():
    """
    Performs a continuous sweep of the quarter-wave plate ellipticity angle to simulate
    the live rotation of the waveplate in the laboratory.
    """
    B_gauss = 0.0
    T = 295.0
    s_pump = 2.0
    kappa_probe = 0.30
    gamma_eff = 12.0
    gamma_tr = 0.15
    N_v = 101

    # Ellipticity angles from -45° (sigma-) to +45° (sigma+)
    eps_angles_deg = np.linspace(-45.0, 45.0, 7)
    cmap = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(eps_angles_deg)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: 87Rb F=1 - Sweep Probe Ellipticity (Pump fixed at +6.6°)
    m87_1 = pm.MANIFOLDS[3]
    nu_eval_87 = np.linspace(-200.0, 350.0, 300)
    ax1 = axes[0]
    eps_pump_fixed = np.radians(6.6)

    for eps_pr_deg, col in zip(eps_angles_deg, cmap):
        eps_pr_rad = np.radians(eps_pr_deg)
        _, _, diff = pm.simulate_manifold_zeeman(
            nu_eval_87,
            m87_1,
            B_gauss,
            eps_pump_fixed,
            eps_pr_rad,
            np.radians(100.0),
            T,
            s_pump,
            kappa_probe,
            gamma_eff,
            gamma_tr,
            N_v,
        )
        ax1.plot(nu_eval_87, diff, label=f"eps_probe = {eps_pr_deg:+.0f}°", color=col, linewidth=1.6)

    ax1.set_title("87Rb F=1: Rotating Probe QWP (Pump fixed eps = +6.6°)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Relative Detuning (MHz)", fontsize=11)
    ax1.set_ylabel("Saturated Absorption Signal (arb)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel 2: 85Rb F=2 - Sweep Pump Ellipticity (Probe fixed at +14.9°)
    m85_2 = pm.MANIFOLDS[2]
    nu_eval_85 = np.linspace(-150.0, 250.0, 300)
    ax2 = axes[1]
    eps_probe_fixed = np.radians(14.9)

    for eps_p_deg, col in zip(eps_angles_deg, cmap):
        eps_p_rad = np.radians(eps_p_deg)
        _, _, diff = pm.simulate_manifold_zeeman(
            nu_eval_85,
            m85_2,
            B_gauss,
            eps_p_rad,
            eps_probe_fixed,
            np.radians(100.0),
            T,
            s_pump,
            kappa_probe,
            gamma_eff,
            gamma_tr,
            N_v,
        )
        ax2.plot(nu_eval_85, diff, label=f"eps_pump = {eps_p_deg:+.0f}°", color=col, linewidth=1.6)

    ax2.set_title("85Rb F=2: Rotating Pump QWP (Probe fixed eps = +14.9°)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Relative Detuning (MHz)", fontsize=11)
    ax2.set_ylabel("Saturated Absorption Signal (arb)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.suptitle(
        "Continuous Quarter-Wave Plate Rotation Sweep on Lower Ground State Manifolds",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    output_sweep_path = os.path.join("../figures", "polarization_qwp_rotation_map.png")
    plt.savefig(output_sweep_path, dpi=200)
    print(f"Saved continuous QWP sweep plot to '{output_sweep_path}'.", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    plot_polarization_effects(200)
    plot_qwp_rotation_sweep()
