import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import pandas as pd


def load(_filepath):
    return pd.read_csv(_filepath).to_numpy().T  # type: ignore


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
    return freq, fabry,pumped,unpumped


def plot_it(fp_name, pumped_name, unpumped_name, cut_edges=False):
    freq,fabry,pumped,unpumped = prep_data(fp_name, pumped_name, unpumped_name, cut_edges)
    diff = pumped-unpumped
    plt.subplot(2, 1, 1)
    plt.plot(freq,pumped, label="Pumped")
    plt.plot(freq,unpumped, label="Unpumped")
    plt.xlabel("F-P peak index")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(freq, diff, label="Diff")
    plt.xlabel("F-P peak index")
    plt.legend()
    plt.show()


def plot_magnetic():
    for i in range(5):
        index = i+1
        pre = "data/Session 9/Magnetic variation/"
        post = ".CSV"
        plot_it(f"{pre}GF{index}{post}",
                f"{pre}GP{index}{post}",
                f"{pre}GU{index}{post}")

plot_magnetic()