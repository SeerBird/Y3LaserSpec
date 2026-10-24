import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import least_squares
import pandas as pd

from util import casimir_energy, MANIFOLDS_CONFIG

show_edges = False
show_freq_fit = False
show_peaks_and_edges = False


def load(_filepath):
    data = pd.read_csv(_filepath).to_numpy().T  # type: ignore
    if len(data) > 2:
        return data[0], data[-1]
    return data


def cut_edges_off(fabry, pumped, unpumped, peaks, peak_index, index):
    linear_freq = np.poly1d(np.polyfit(peaks, peak_index, 1))
    edge_indices = find_peaks(peak_index - linear_freq(peaks))[0]
    assert len(edge_indices) == 2
    # go inwards a bit
    edge_indices[0] += 1
    edge_indices[1] -= 1
    left_edge, right_edge = peaks[edge_indices]
    peaks = peaks[edge_indices[0]:edge_indices[1] + 1]
    peak_index = peak_index[edge_indices[0]:edge_indices[1] + 1]
    index = index[left_edge:right_edge + 1]
    if show_peaks_and_edges:
        plt.subplot(2, 2, 1)
        plt.plot(fabry)
        plt.plot(peaks, fabry[peaks], "rx")
        plt.subplot(2, 2, 2)
        plt.plot(peaks, peak_index, "x", label="Peaks")
        plt.plot(index, linear_freq(index), label="Linear fit")
        plt.subplot(2, 2, 3)
        plt.plot(peaks, peak_index - linear_freq(peaks), "x", label="Diff")
        plt.subplot(2, 2, 4)
        plt.plot(pumped)
        plt.axvline(left_edge, color='r', label="Edges")
        plt.axvline(right_edge, color='r')
        plt.show()
    return (fabry[left_edge:right_edge + 1],
            unpumped[left_edge:right_edge + 1],
            pumped[left_edge:right_edge + 1], peaks, peak_index, index)


def prep_data(fp_name, pumped_name, unpumped_name, cut_edges=False):
    _, fabry = load(fp_name)
    _, pumped = load(pumped_name)
    _, unpumped = load(unpumped_name)
    peaks, _ = find_peaks(fabry, prominence=1e-2)
    index = np.arange(len(fabry))
    peak_index = np.arange(len(peaks))
    if cut_edges:
        fabry, pumped, unpumped, peaks, peak_index, index = cut_edges_off(fabry, pumped, unpumped,
                                                                          peaks, peak_index, index)
    freq_fit = np.poly1d(np.polyfit(peaks, peak_index, 5))
    freq = freq_fit(index)
    if show_freq_fit:
        plt.subplot(2, 2, 1)
        plt.plot(freq, pumped)
        plt.subplot(2, 2, 2)
        plt.plot(index, freq)
        plt.plot(peaks, peak_index)
        plt.subplot(2, 2, 3)
        plt.plot(peaks, peak_index - freq_fit(peaks))
        plt.subplot(2, 2, 4)
        plt.plot(freq, pumped - unpumped)
        plt.show()
    return freq, fabry, pumped, unpumped


# region hyperfine multiplet fitting



DEFAULT_WINDOWS = [(9.0, 10.6), (12.6, 13.6), (20.6, 21.4), (27.1, 28.1)]


def calc_peak_offsets(A, B,Fs):
    E0 = casimir_energy(F0, I, J, A, B)
    E1 = casimir_energy(F1, I, J, A, B)
    E2 = casimir_energy(F2, I, J, A, B)
    Es = [E0, E1, E2]
    E_ref = Es[ref_idx]
    return np.array([
        E0 - E_ref,
        0.5 * (E0 + E1) - E_ref,
        E1 - E_ref,
        0.5 * (E0 + E2) - E_ref,
        0.5 * (E1 + E2) - E_ref,
        E2 - E_ref
    ])

def fit_hyperfine_multiplets(freq, diff, windows=DEFAULT_WINDOWS, fsr=374.74):
    """
    Fits each hyperfine manifold with a set of 6 Lorentzians whose relative positions
    are mathematically constrained by Casimir's formula and crossover resonance midpoints.
    
    Amplitudes are bounded >= 0, so missing peaks (such as F'=0) naturally converge to 0
    without requiring boolean flags or destabilizing the fit.
    
    Parameters
    ----------
    freq : ndarray
        Fabry-Pérot fringe index axis from prep_data.
    diff : ndarray
        Differential absorption signal (pumped - unpumped).
    windows : list of (float, float), optional
        Fringe range (min, max) for each of the 4 manifolds.
    fsr : float, optional
        Fabry-Pérot cavity Free Spectral Range in MHz/fringe (nominal 374.74 MHz).
        
    Returns
    -------
    list of dict
        Fitted parameters for each manifold:
        'name', 'A', 'B', 'gamma', 'f0_fringe', 'f0_mhz', 'amplitudes',
        'peak_freqs', 'peak_names', 'x_fit', 'y_fit', 'x_data', 'y_data'.
    """
    results = []
    nu_total = freq * fsr

    for cfg, (w_min, w_max) in zip(MANIFOLDS_CONFIG, windows):
        mask = (freq >= w_min) & (freq <= w_max)
        x_mhz = nu_total[mask]
        y_data = diff[mask]
        if len(x_mhz) == 0:
            continue

        I, J = cfg.I, cfg.J
        F0, F1, F2 = cfg.Fp_levels
        ref_idx = cfg.ref_index

        def calc_peak_offsets(A, B):
            E0 = casimir_energy(F0, I, J, A, B)
            E1 = casimir_energy(F1, I, J, A, B)
            E2 = casimir_energy(F2, I, J, A, B)
            Es = [E0, E1, E2]
            E_ref = Es[ref_idx]
            return np.array([
                E0 - E_ref,
                0.5 * (E0 + E1) - E_ref,
                E1 - E_ref,
                0.5 * (E0 + E2) - E_ref,
                0.5 * (E1 + E2) - E_ref,
                E2 - E_ref
            ])

        def evaluate_model(p, x_eval):
            nu_ref, A, B, gamma, bg = p[:5]
            amps = p[5:]
            offsets = calc_peak_offsets(A, B)
            y_eval = np.full_like(x_eval, bg)
            for k in range(6):
                y_eval += amps[k] / (1.0 + 4.0 * ((x_eval - (nu_ref + offsets[k])) / gamma) ** 2)
            return y_eval

        def residuals(p):
            return evaluate_model(p, x_mhz) - y_data

        # Initialize reference frequency near the appropriate edge
        nu_ref_init = (
                          0.8 * w_max + 0.2 * w_min if ref_idx == 2 else 0.2 * w_max + 0.8 * w_min) * fsr
        A_init, B_init = cfg.ABhfs_guess
        bg_init = float(np.median(y_data))
        p0 = np.array([nu_ref_init, A_init, B_init, 15.0, bg_init] + [0.02] * 6, dtype=float)

        bg_min = min(-2.0, float(np.min(y_data)) - 1.0)
        bg_max = max(2.0, float(np.max(y_data)) + 1.0)
        lower = np.array([w_min * fsr, 10.0, 0.0, 3.0, bg_min] + [0.0] * 6, dtype=float)
        upper = np.array([w_max * fsr, 120.0, 50.0, 40.0, bg_max] + [2.0] * 6, dtype=float)

        p0 = np.clip(p0, lower + 1e-4, upper - 1e-4)

        fit = least_squares(residuals, p0, bounds=(lower, upper), ftol=1e-4, xtol=1e-4)
        nu_ref, A_fit, B_fit, gamma_fit, bg_fit = fit.x[:5]
        amps_fit = fit.x[5:]
        peak_offsets = calc_peak_offsets(A_fit, B_fit)

        x_dense = np.linspace(x_mhz[0], x_mhz[-1], 300)
        y_dense = evaluate_model(fit.x, x_dense)

        results.append({
            'name': cfg['name'],
            'A': float(A_fit),
            'B': float(B_fit),
            'gamma': float(gamma_fit),
            'f0_mhz': float(nu_ref),
            'f0_fringe': float(nu_ref / fsr),
            'bg': float(bg_fit),
            'amplitudes': amps_fit,
            'peak_freqs': nu_ref + peak_offsets,
            'peak_names': cfg['names'],
            'x_data': freq[mask],
            'y_data': y_data,
            'x_fit': x_dense / fsr,
            'y_fit': y_dense,
            'cost': float(fit.cost),
        })

    return results


# Alias for convenience / alternate spelling
fit_hyperfine_multipliers = fit_hyperfine_multiplets


def plot_hyperfine_fits(results, title="Hyperfine Multiplet Casimir Fits"):
    """Plots the fitted Casimir Lorentzians against measured data for all manifolds."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plt.suptitle(title, fontsize=13, fontweight='bold')
    for i, res in enumerate(results):
        ax = axs.flat[i]
        ax.plot(res['x_data'], res['y_data'], label='Measured Diff', color='navy', alpha=0.85)
        ax.plot(res['x_fit'], res['y_fit'], '--', label='Casimir Fit', color='crimson',
                linewidth=2.0)
        ax.set_title(
            f"{res['name']}: A = {res['A']:.2f} MHz, B = {res['B']:.2f} MHz ($\\Gamma$ = {res['gamma']:.1f} MHz)",
            fontsize=11, fontweight='bold'
        )
        ax.set_xlabel("F-P Fringe Index")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# endregion


# region plot stuff to check
def plot_it(fp_name, pumped_name, unpumped_name, cut_edges=False, title=""):
    freq, fabry, pumped, unpumped = prep_data(fp_name, pumped_name, unpumped_name, cut_edges)
    diff = pumped - unpumped
    plt.suptitle(title)
    plt.subplot(2, 1, 1)
    plt.plot(freq, pumped, label="Pumped")
    plt.plot(freq, unpumped, label="Unpumped")
    plt.xlabel("F-P peak index")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(freq, diff, label="Diff")
    plt.xlabel("F-P peak index")
    plt.legend()
    plt.show()


def plot_magnetic():
    fields = [19, 39, 59, 79, 98, 119]
    for i in range(5):
        index = i + 1
        pre = "data/Session 9/Magnetic variation/"
        post = ".CSV"
        plot_it(f"{pre}GF{index}{post}",
                f"{pre}GP{index}{post}",
                f"{pre}GU{index}{post}",
                title=f"{fields[i]} mA")

# endregion
