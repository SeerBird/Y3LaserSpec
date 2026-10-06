from pathlib import Path
from scipy.constants import c,k,u
import pandas as pd, numpy as np, matplotlib.pyplot as plt

what_is_this, c2, spectrum = pd.read_csv("data//DOPPWA02.CSV").to_numpy().T

index = np.arange(len(spectrum))
fine_freq = 384.23e12
first_peak_nu = fine_freq -2.563e9
last_peak_nu = fine_freq + 4.272e9
first_peak = 21300
last_peak = 88000
conversion_factor = (last_peak_nu - first_peak_nu) / (last_peak - first_peak)
freq = (first_peak_nu + conversion_factor
        * (index - first_peak))
FWHM = 6500 * conversion_factor
T = (FWHM * c/fine_freq)**2/(8*k*np.log(2))*85.4678*u
print(f"Temp: {T} °K")

plt.plot(freq-fine_freq,spectrum, label="c3")
plt.legend()
plt.show()
