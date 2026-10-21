from pathlib import Path

from scipy import fft
from scipy.constants import c, k, u
from scipy.signal import find_peaks
import pandas as pd, numpy as np, matplotlib.pyplot as plt

show_data = True
check_magic = True
show_peaks_and_edges = True
show_freq_fit = True


# region load data
def load(_filepath):
    return pd.read_csv(_filepath).to_numpy().T  # type: ignore


_, fabry = load("data//Session 7//FP9.CSV")
_, c3_pumped = load("data//Session 7//C3P9.CSV")
_, c3_unpumped = load("data//Session 7//C3U9.CSV")
diff_raw = c3_pumped - c3_unpumped
if show_data:
    plt.subplot(2, 1, 1)
    plt.plot(c3_pumped, label="Pumped")
    plt.plot(c3_unpumped, label="Unpumped")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(diff_raw, label="Diff")
    plt.legend()
    plt.show()
# endregion
# region check the weird antiabsorption in our first measurements
if check_magic:
    _, _, dips = load("data//DIPS3.CSV")
    _, _, dips1 = load("data//NOPUMP4.CSV")
    plt.plot(dips, label="Pumped")
    plt.plot(dips1, label="Unpumped")
    plt.show()
# endregion
# region temp estimate
fine_freq = 384.23e12
first_peak_nu = fine_freq - 2.563e9
last_peak_nu = fine_freq + 4.272e9
first_peak = 21300
last_peak = 88000
conversion_factor = (last_peak_nu - first_peak_nu) / (last_peak - first_peak)
FWHM = 6500 * conversion_factor
T = (FWHM * c / fine_freq) ** 2 / (8 * k * np.log(2)) * 85.4678 * u
print(f"Temp: {T} °K")
# endregion
# region get peaks and edges
peaks, _ = find_peaks(fabry, prominence=1e-2)
index = np.arange(len(fabry))
peak_index = np.arange(len(peaks))
linear_freq = np.poly1d(np.polyfit(peaks, peak_index, 1))
edge_indices = find_peaks(peak_index - linear_freq(peaks))[0]
assert len(edge_indices) == 2
# go inwards a bit
edge_indices[0] += 1
edge_indices[1] -= 1

left_edge, right_edge = peaks[edge_indices]
if show_peaks_and_edges:
    plt.subplot(2, 2, 1)
    plt.plot(fabry)
    plt.plot(peaks, fabry[peaks], "rx")
    plt.subplot(2, 2, 2)
    plt.plot(peaks, peak_index,"x", label="Peaks")
    plt.plot(index, linear_freq(index), label="Linear fit")
    plt.subplot(2, 2, 3)
    plt.plot(peaks, peak_index - linear_freq(peaks),"x", label="Diff")
    plt.subplot(2, 2, 4)
    plt.plot(c3_pumped)
    plt.axvline(left_edge, color='r', label="Edges")
    plt.axvline(right_edge, color='r')
    plt.show()
# endregion
# region cut data to the usable slice, get frequency axis
peaks = peaks[edge_indices[0]:edge_indices[1] + 1]
peak_index = peak_index[edge_indices[0]:edge_indices[1] + 1]
index = index[left_edge:right_edge + 1]
fabry = fabry[left_edge:right_edge + 1]
c3_pumped = c3_pumped[left_edge:right_edge + 1]
c3_unpumped = c3_unpumped[left_edge:right_edge + 1]
diff = c3_pumped - c3_unpumped

freq_fit = np.poly1d(np.polyfit(peaks, peak_index, 5))
freq = freq_fit(index)
if show_freq_fit:
    plt.subplot(2, 2, 1)
    plt.plot(freq, c3_pumped)
    plt.subplot(2,2, 2)
    plt.plot(index, freq)
    plt.plot(peaks, peak_index)
    plt.subplot(2,2, 3)
    plt.plot(peaks, peak_index - freq_fit(peaks))
    plt.subplot(2, 2, 4)
    plt.plot(freq, diff)
    plt.show()
# endregion
