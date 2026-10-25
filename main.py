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


def calc_peak_offsets(A, B, I, J, F):
    F = np.asarray(F, dtype=float)
    K = F * (F + 1) - I * (I + 1) - J * (J + 1)
    if I > 0.5 and J > 0.5:
        denom = 4.0 * I * (2.0 * I - 1.0) * J * (2.0 * J - 1.0)
        Bterm = B * (1.5 * K * (K + 1) - 2.0 * I * (I + 1) * J * (J + 1)) / denom
    else:
        Bterm = 0.0
    E = 0.5 * A * K + Bterm
    i, j = np.triu_indices(len(E), k=1)
    E_co = 0.5 * (E[i] + E[j])
    return np.concatenate([E, E_co])


DEFAULT_WINDOWS = [(9.0, 10.6), (12.6, 13.6), (20.6, 21.4), (27.1, 28.1)]


def fit_hyperfine_multiplets(freq, diff, windows=None, fsr=374.74):
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
    if windows is None:
        windows = DEFAULT_WINDOWS
    results = []
    nu_total = freq * fsr

    for cfg, (w_min, w_max) in zip(MANIFOLDS_CONFIG, windows):
        mask = (freq >= w_min) & (freq <= w_max)
        x_mhz = nu_total[mask]
        y_data = diff[mask]
        if len(x_mhz) == 0:
            continue

        ref_idx = cfg.ref_index

        def evaluate_model(p, x_eval):
            nu_ref, A, B, gamma, bg = p[:5]
            amps = p[5:]
            y_eval = np.full_like(x_eval, bg)
            E_all = calc_peak_offsets(A, B, cfg.I, cfg.J, cfg.Fp_levels)
            offsets = E_all - E_all[ref_idx]
            for k in range(len(offsets)):
                y_eval += amps[k] / (1.0 + 4.0 * ((x_eval - (nu_ref + offsets[k])) / gamma) ** 2)
            return y_eval

        def residuals(p):
            return evaluate_model(p, x_mhz) - y_data

        # Initial reference frequency guess aligned to prominent data peaks
        A_init, B_init = cfg.ABhfs_guess
        E_init = calc_peak_offsets(A_init, B_init, cfg.I, cfg.J, cfg.Fp_levels)
        offsets_init = E_init - E_init[ref_idx]

        dpeaks, _ = find_peaks(y_data, prominence=0.01)
        if len(dpeaks) > 0:
            if cfg.name == '85Rb F=2':
                # In 85Rb F=2, the dominant peak is CO 2-3 (index 5, ~61.1 MHz above F'=1)
                nu_ref_init = float(x_mhz[dpeaks[-1]] - offsets_init[5])
            elif cfg.name == '87Rb F=1':
                # In 87Rb F=1, the rightmost peak is F'=2 (index 2, ~229.1 MHz above F'=0)
                nu_ref_init = float(x_mhz[dpeaks[-1]] - offsets_init[2])
            else:
                k_max = np.argmax(offsets_init)
                nu_ref_init = float(x_mhz[dpeaks[-1]] - offsets_init[k_max])
        else:
            nu_ref_init = float(0.5 * (x_mhz[0] + x_mhz[-1]))

        bg_init = float(np.percentile(y_data, 10))
        amp_init = float(np.max(y_data) - bg_init) / 3.0
        p0 = np.array([nu_ref_init, A_init, B_init, 20.0, bg_init] + [amp_init] * 6, dtype=float)

        lower = np.array([w_min * fsr - 100.0, 5.0, 0.0, 8.0, -1.0] + [0.0] * 6, dtype=float)
        upper = np.array([w_max * fsr + 100.0, 150.0, 60.0, 32.0, 2.0] + [2.0] * 6, dtype=float)

        p0 = np.clip(p0, lower + 1e-4, upper - 1e-4)

        fit = least_squares(residuals, p0, bounds=(lower, upper), ftol=1e-5, xtol=1e-5)
        nu_ref, A_fit, B_fit, gamma_fit, bg_fit = fit.x[:5]
        amps_fit = fit.x[5:]
        E_all_fit = calc_peak_offsets(A_fit, B_fit, cfg.I, cfg.J, cfg.Fp_levels)
        peak_offsets = E_all_fit - E_all_fit[ref_idx]

        x_dense = np.linspace(x_mhz[0], x_mhz[-1], 300)
        y_dense = evaluate_model(fit.x, x_dense)
        bg_dense = np.full_like(x_dense, bg_fit)

        results.append({
            'name': cfg.name,
            'A': float(A_fit),
            'B': float(B_fit),
            'gamma': float(gamma_fit),
            'f0_mhz': float(nu_ref),
            'f0_fringe': float(nu_ref / fsr),
            'bg': float(bg_fit),
            'bg_dense': bg_dense,
            'amplitudes': amps_fit,
            'peak_freqs': nu_ref + peak_offsets,
            'peak_names': cfg.names,
            'x_data': x_mhz,
            'y_data': y_data,
            'x_fit': x_dense,
            'y_fit': y_dense,
            'cost': float(fit.cost),
        })

    return results


# endregion


# region plot stuff to check
def plot_it(fp_name, pumped_name, unpumped_name, cut_edges=False, title="", fsr=374.74):
    freq, fabry, pumped, unpumped = prep_data(fp_name, pumped_name, unpumped_name, cut_edges)
    diff = pumped - unpumped
    freq_mhz = freq * fsr
    plt.suptitle(title)
    plt.subplot(2, 1, 1)
    plt.plot(freq_mhz, pumped, label="Pumped")
    plt.plot(freq_mhz, unpumped, label="Unpumped")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(freq_mhz, diff, label="Diff")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Voltage (V)")
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


def plot_hyperfine_fits(results, title="Hyperfine Multiplet Casimir Fits", save_path="figures/hyperfine_multiplets_fits.png"):
    """
    Plots the fitted Casimir Lorentzians against measured data for all manifolds,
    including individual contributing Lorentzian components on the same axes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plt.suptitle(title, fontsize=13, fontweight='bold')
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:olive']

    for i, res in enumerate(results):
        ax = axs.flat[i]
        ax.plot(res['x_data'], res['y_data'], label='Measured Diff', color='black', alpha=0.6, linewidth=1.2)
        ax.plot(res['x_fit'], res['y_fit'], color='crimson', linewidth=2.0, label='Total Casimir Fit')
        
        x_dense = res['x_fit']
        bg_dense = res.get('bg_dense', np.full_like(x_dense, res.get('bg', 0.0)))
        ax.plot(x_dense, bg_dense, color='gray', linestyle=':', linewidth=1.5, label='Background Offset')

        gamma = res['gamma']
        peak_names = res.get('peak_names', [])

        for k in range(len(res['amplitudes'])):
            amp = res['amplitudes'][k]
            nu_k = res['peak_freqs'][k]
            y_comp = bg_dense + amp / (1.0 + 4.0 * ((x_dense - nu_k) / gamma) ** 2)
            name_k = peak_names[k] if k < len(peak_names) else f"Peak {k+1}"
            if amp > 1e-4:
                color_k = colors[k % len(colors)]
                ax.plot(x_dense, y_comp, '--', color=color_k, linewidth=1.2, label=name_k)
                ax.fill_between(x_dense, bg_dense, y_comp, color=color_k, alpha=0.08)

        ax.set_title(
            f"{res['name']}: A = {res['A']:.2f} MHz, B = {res['B']:.2f} MHz ($\\Gamma$ = {res['gamma']:.1f} MHz)",
            fontsize=11, fontweight='bold'
        )
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
# endregion

if __name__ == "__main__":
    freq, fabry, pumped, unpumped = prep_data("data/Session 9/F1.CSV",
                                              "data/Session 9/P1.CSV",
                                              "data/Session 9/U1.CSV")
    diff = pumped - unpumped
    windows_s9 = [(3.8, 5.3), (7.2, 8.5), (15.0, 16.2), (21.5, 22.8)]
    results = fit_hyperfine_multiplets(freq, diff, windows=windows_s9)
    for res in results:
        print(f"{res['name']}: A = {res['A']:.2f} MHz, B = {res['B']:.2f} MHz, Gamma = {res['gamma']:.1f} MHz, cost = {res['cost']:.4f}")
    plot_hyperfine_fits(results)