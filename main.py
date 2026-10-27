import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import least_squares, minimize_scalar
from scipy.interpolate import interp1d
import pandas as pd

from util import casimir_energy, calc_peak_offsets, MANIFOLDS_CONFIG

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


def prep_data(fp_name, pumped_name, unpumped_name, cut_edges=False, align_shift=True):
    _, fabry = load(fp_name)
    _, pumped = load(pumped_name)
    _, unpumped = load(unpumped_name)
    peaks, _ = find_peaks(fabry, prominence=1e-2)
    index = np.arange(len(fabry))
    peak_index = np.arange(len(peaks))
    if cut_edges:
        fabry, pumped, unpumped, peaks, peak_index, index = cut_edges_off(fabry, pumped, unpumped,
                                                                          peaks, peak_index, index)
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
            bounds=(-150, 150),
            method='bounded'
        )
        unpumped = u_interp(index + res.x)

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


# region Cycling Line Analysis Pipeline


def load_and_calibrate(fp_file, pump_file, unpump_file, fsr=374.74, align_shift=True):
    """
    Convenience wrapper around prep_data.
    Loads raw oscilloscope traces, applies shift alignment, fits Fabry-Perot fringes,
    and returns:
        t : oscilloscope time axis (seconds)
        nu : calibrated frequency axis (MHz)
        diff : differential absorption signal (pumped - unpumped) in mV
    """
    t, _ = load(pump_file)
    freq, fabry, pumped, unpumped = prep_data(fp_file, pump_file, unpump_file, align_shift=align_shift)
    nu = freq * fsr
    diff = (pumped - unpumped) * 1e3  # in mV
    return t, nu, diff


def select_cycling_peak(t, nu, diff, isotope='87', span_mhz=20.0):
    """
    Selects and isolates the cycling transition line within a calibrated spectrum.

    For 87Rb:
        The F=2 -> F'=3 cycling transition is located at the blue end of the F=2 manifold,
        isolated from the nearest crossover resonance CO(2,3) by ~66.7 MHz.
    For 85Rb:
        The F=3 -> F'=4 cycling transition is located exactly +60.32 MHz above the dominant
        CO(3,4) crossover peak.

    Parameters:
    -----------
    t : ndarray
        Oscilloscope time axis (seconds).
    nu : ndarray
        Calibrated frequency axis (MHz).
    diff : ndarray
        Differential absorption signal (mV).
    isotope : str
        '87' for 87Rb (F=2 -> F'=3) or '85' for 85Rb (F=3 -> F'=4).
    span_mhz : float
        Half-width of frequency window to extract around the peak (MHz).

    Returns:
    --------
    dict containing:
        'x': detuning from peak center (MHz, centered at 0)
        'y': differential signal in mV within the window
        'nu_center': absolute frequency at peak center (MHz)
        't_center': time at peak center (seconds)
        'isotope': isotope label ('87' or '85')
        'transition': formatted transition name string
        'span_mhz': half-window span in MHz
    """
    if isotope == '87':
        # 87Rb F=2 -> F'=3 cycling peak located in time window [-0.0036, -0.0031] s
        m = (t >= -0.0036) & (t <= -0.0031)
        if not np.any(m):
            raise ValueError("Time window for 87Rb cycling peak not found in data.")
        t_ref = t[m][np.argmax(diff[m])]
        nu_ref = nu[np.argmin(np.abs(t - t_ref))]
        transition_name = "87Rb (F=2 -> F'=3)"
    elif isotope == '85':
        # 85Rb F=3 -> F'=4 cycling peak anchored +60.32 MHz above prominent CO(3,4) peak
        m_co = (t >= 0.00030) & (t <= 0.00060)
        if not np.any(m_co):
            raise ValueError("Time window for 85Rb CO(3,4) peak not found in data.")
        t_co = t[m_co][np.argmax(diff[m_co])]
        nu_co = nu[np.argmin(np.abs(t - t_co))]
        nu_ref = nu_co + 60.32
        t_ref = t[np.argmin(np.abs(nu - nu_ref))]
        transition_name = "85Rb (F=3 -> F'=4)"
    else:
        raise ValueError(f"Unknown isotope '{isotope}'. Choose '87' or '85'.")

    # Refine peak center to local maximum within +/- 3 MHz of the nominal position
    m_fine = np.abs(nu - nu_ref) <= 3.0
    if np.any(m_fine):
        idx_peak = np.where(m_fine)[0][np.argmax(diff[m_fine])]
        nu_center = nu[idx_peak]
        t_center = t[idx_peak]
    else:
        nu_center = nu_ref
        t_center = t_ref

    # Extract isolated window
    mask = np.abs(nu - nu_center) <= span_mhz
    x = nu[mask] - nu_center
    y = diff[mask]

    return {
        'x': x,
        'y': y,
        'nu_center': float(nu_center),
        't_center': float(t_center),
        'isotope': isotope,
        'transition': transition_name,
        'span_mhz': span_mhz
    }


def measure_fwhm_filtered_crossing(x, y, window_length=31, polyorder=3, wing_margin=None):
    """
    Extracts the Full Width at Half Maximum (FWHM) of an isolated spectral peak
    using the noise-filtered half-maximum crossing method.

    Algorithm:
    1. Baseline Estimation:
       Fits a local linear baseline y_bg(x) = c0 + c1*x to the outer wings (|x| >= wing_margin).
    2. Subtraction:
       Subtracts the baseline to obtain the net peak signal: y_net = y - y_bg.
    3. Noise Filtering:
       Applies a zero-phase Savitzky-Golay filter to suppress high-frequency noise
       while preserving peak curvature (broadening on a 6 MHz peak is < 0.008 MHz).
    4. Half-Maximum Determination:
       Determines peak summit amplitude y_max = max(y_smooth) at x_max,
       and sets the half-maximum threshold hm = 0.5 * y_max.
    5. Monotonic Edge Crossings:
       Searches outward from x_max toward the wings:
       - Left crossing (x <= x_max): first point dropping below hm, linearly interpolated.
       - Right crossing (x >= x_max): first point dropping below hm, linearly interpolated.
    6. FWHM:
       FWHM = x_right - x_left.

    Parameters:
    -----------
    x : ndarray
        Detuning axis in MHz (centered near 0).
    y : ndarray
        Differential absorption signal in mV.
    window_length : int
        Savitzky-Golay filter window size in points (must be odd, default: 31 points ~ 1.2 MHz).
    polyorder : int
        Savitzky-Golay polynomial order (default: 3).
    wing_margin : float, optional
        Detuning threshold beyond which points are used for baseline estimation (MHz).
        Defaults to 0.6 * max(|x|).

    Returns:
    --------
    dict containing:
        'fwhm': measured FWHM in MHz
        'x_left': left crossing frequency in MHz
        'x_right': right crossing frequency in MHz
        'hm_level': half-maximum voltage level (mV above baseline)
        'y_max': peak amplitude (mV above baseline)
        'x_max': detuning at peak maximum (MHz)
        'y_smooth': filtered net signal (mV)
        'y_bg': estimated linear baseline (mV)
        'c0': baseline offset (mV)
        'c1': baseline slope (mV/MHz)
        'x': detuning array (MHz)
        'y': input differential signal (mV)
    """
    if wing_margin is None:
        wing_margin = 0.6 * np.max(np.abs(x))

    wing_mask = np.abs(x) >= wing_margin
    if np.sum(wing_mask) > 10:
        c1, c0 = np.polyfit(x[wing_mask], y[wing_mask], 1)
    else:
        c0, c1 = float(np.min(y)), 0.0
    y_bg = c0 + c1 * x
    y_net = y - y_bg

    # Ensure window_length is odd and does not exceed array length
    w_len = min(window_length, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if w_len < polyorder + 2:
        w_len = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    y_smooth = savgol_filter(y_net, window_length=w_len, polyorder=polyorder)

    idx_max = np.argmax(y_smooth)
    x_max = x[idx_max]
    y_max = y_smooth[idx_max]
    hm = 0.5 * y_max

    # Find left crossing: walk leftwards from idx_max
    left_mask = (x <= x_max) & (y_smooth <= hm)
    left_indices = np.where(left_mask)[0]
    if len(left_indices) == 0:
        raise ValueError("Could not find left half-maximum crossing.")
    i1 = left_indices[-1]
    i2 = i1 + 1
    # Linear sub-sample interpolation
    x_left = x[i1] + (hm - y_smooth[i1]) * (x[i2] - x[i1]) / (y_smooth[i2] - y_smooth[i1])

    # Find right crossing: walk rightwards from idx_max
    right_mask = (x >= x_max) & (y_smooth <= hm)
    right_indices = np.where(right_mask)[0]
    if len(right_indices) == 0:
        raise ValueError("Could not find right half-maximum crossing.")
    j2 = right_indices[0]
    j1 = j2 - 1
    # Linear sub-sample interpolation
    x_right = x[j1] + (hm - y_smooth[j1]) * (x[j2] - x[j1]) / (y_smooth[j2] - y_smooth[j1])

    fwhm = float(x_right - x_left)

    return {
        'fwhm': fwhm,
        'x_left': float(x_left),
        'x_right': float(x_right),
        'hm_level': float(hm),
        'y_max': float(y_max),
        'x_max': float(x_max),
        'y_smooth': y_smooth,
        'y_bg': y_bg,
        'c0': float(c0),
        'c1': float(c1),
        'x': x,
        'y': y
    }


# endregion


if __name__ == '__main__':
    # Quick self-test demonstration on low-power Run 10
    print("Testing cycling line analysis pipeline on Session 10 Run 10...")
    t, nu, diff = load_and_calibrate(
        'data/Session 10/Power variation/F10.CSV',
        'data/Session 10/Power variation/P10.CSV',
        'data/Session 10/Power variation/U10.CSV'
    )
    peak87 = select_cycling_peak(t, nu, diff, isotope='87', span_mhz=20.0)
    result87 = measure_fwhm_filtered_crossing(peak87['x'], peak87['y'], wing_margin=12.0)
    print(f"87Rb Cycling Line FWHM: {result87['fwhm']:.2f} MHz (Amplitude: {result87['y_max']:.2f} mV)")

    peak85 = select_cycling_peak(t, nu, diff, isotope='85', span_mhz=18.0)
    result85 = measure_fwhm_filtered_crossing(peak85['x'], peak85['y'], wing_margin=10.0)
    print(f"85Rb Cycling Line FWHM: {result85['fwhm']:.2f} MHz (Amplitude: {result85['y_max']:.2f} mV)")




