import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def power_at_angle(theta_deg, Ex, Ey, delta):
    """
    Calculates transmitted power through a linear polarizer at angle theta_deg
    for an elliptical beam with linear components Ex, Ey and relative phase delta (radians).
    
    P(theta) = (Ex*cos(theta))^2 + (Ey*sin(theta))^2 + 2*Ex*Ey*cos(theta)*sin(theta)*cos(delta)
    """
    th = np.radians(theta_deg)  # TODO: do units
    return (Ex * np.cos(th)) ** 2 + (Ey * np.sin(th)) ** 2 + 2.0 * Ex * Ey * np.cos(th) * np.sin(
        th) * np.cos(delta)


def fit_polarization(angles_deg, powers):
    """
    Fits (Ex, Ey, delta) to measured power as a function of polarizer angle in degrees.
    Returns (Ex, Ey, delta_rad).
    """
    angles = np.asarray(angles_deg, dtype=float)
    p = np.asarray(powers, dtype=float)

    p0 = [np.sqrt(np.max(p)), np.sqrt(np.min(p)), 0.0]
    popt, _ = curve_fit(
        power_at_angle,
        angles,
        p,
        p0=p0,
        bounds=([0.0, 0.0, -np.pi], [np.inf, np.inf, np.pi]),
    )
    return popt[0], popt[1], popt[2]


def get_ellipticity(Ex, Ey, delta):
    """
    Calculates the ellipticity angle epsilon in degrees from (Ex, Ey, delta).
    sin(2*eps) = S3 / S0 = 2*Ex*Ey*sin(delta) / (Ex^2 + Ey^2).
    0° = linear polarization, 45° = circular polarization.
    """
    s = (2.0 * Ex * Ey * np.sin(delta)) / (Ex ** 2 + Ey ** 2)
    return float(np.degrees(0.5 * np.arcsin(np.clip(s, -1.0, 1.0))))


def analyze_datasets(filepath, invert_indices,names):
    _data = pd.read_excel(filepath).to_numpy().T
    angles_data = _data[0]
    _data = _data[1:]
    fineangles = np.linspace(np.min(angles_data), np.max(angles_data), 500)
    for i, dataset in enumerate(_data):
        if i in invert_indices:
            angles = 180 - angles_data
        else:
            angles = angles_data
        fit = fit_polarization(angles, dataset)
        axis = np.angle(fit[0]+1j*fit[1],deg = True)
        plt.subplot(2, 2, i + 1)
        plt.plot(angles, dataset, "x", label="Data")
        plt.plot(fineangles, power_at_angle(fineangles, *fit), label="Fit")
        title = f"{names[i]}|Ex:{fit[0]:.2F},Ey:{fit[1]:.2F},λ:{axis:.2F},ε:{get_ellipticity(*fit):.2F}"
        print(title)
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_datasets("data/Clean polarizations.xlsx",
                     (1, 3),
                     ("probe", "pump", "probe_tied", "pump_again"))
    analyze_datasets("data/Incoming polarization.xlsx",[3],
                     ["front","between","probe","pump"])
