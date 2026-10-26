import os
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.optimize import least_squares
import main

fsr = 374.74

# Casimir literature values
E87_all = main.calc_peak_offsets(84.7185, 12.4965, 1.5, 1.5, [1, 2, 3])
offsets_87 = E87_all - E87_all[2] # rel to F'=3

E85_all = main.calc_peak_offsets(25.0020, 25.7900, 2.5, 1.5, [2, 3, 4])
offsets_85 = E85_all - E85_all[2] # rel to F'=4

df_pwr = pd.read_excel('data/Session 10/Power variation/Powers.xlsx')
powers = df_pwr['Power_uW'].values

def load_and_calibrate(r_id):
    df_p = pd.read_csv(f'data/Session 10/Power variation/P{r_id}.CSV')
    df_u = pd.read_csv(f'data/Session 10/Power variation/U{r_id}.CSV')
    f_path = f'data/Session 10/Power variation/F{r_id}.CSV'
    if not os.path.exists(f_path):
        f_path = 'data/Session 10/Power variation/F14.CSV'
    df_f = pd.read_csv(f_path)
    t = df_p.iloc[:, 0].values
    diff = df_p.iloc[:, 1].values - df_u.iloc[:, 1].values
    fp = df_f.iloc[:, 1].values
    peaks_fp, _ = signal.find_peaks(fp, height=-0.045, prominence=0.005, distance=1000)
    poly = np.poly1d(np.polyfit(t[peaks_fp], np.arange(len(peaks_fp)), 4))
    nu = poly(t) * fsr
    return t, nu, diff

def fit_manifold(nu_man, diff_man, offsets, ref_guess):
    def model_fn(p, x):
        nu_ref, c0, c1 = p[:3]
        gammas = p[3:9]
        amps = p[9:15]
        y = c0 + c1 * (x - nu_ref)
        for k in range(6):
            y += amps[k] / (1.0 + 4.0 * ((x - (nu_ref + offsets[k])) / gammas[k])**2)
        return y
    p0 = [ref_guess, float(np.percentile(diff_man, 5)), 0.0] + [10.0]*6 + [float(np.max(diff_man))/3.0]*6
    b_low = [ref_guess - 40, -0.05, -0.001] + [2.0]*6 + [0.0]*6
    b_high = [ref_guess + 40, 0.05, 0.001] + [45.0]*6 + [1.0]*6
    fit = least_squares(lambda p: model_fn(p, nu_man) - diff_man, p0, bounds=(b_low, b_high), ftol=1e-7, xtol=1e-7)
    
    # Uncertainties
    dof = len(diff_man) - len(p0)
    s_sq = 2.0 * fit.cost / max(1, dof)
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * s_sq
        unc = np.sqrt(np.diagonal(cov))
    except:
        unc = np.zeros_like(fit.x)
        
    res_rms = np.std(diff_man - model_fn(fit.x, nu_man)) * 1e3
    return fit.x, unc, res_rms

rows = []
for r_id, pwr in enumerate(powers, 1):
    t, nu, diff = load_and_calibrate(r_id)
    
    # 87Rb
    m87_loc = (t >= -0.0035) & (t <= -0.0031)
    t87 = t[m87_loc][np.argmax(diff[m87_loc])]
    nu87_ref = nu[np.argmin(np.abs(t - t87))]
    mask87 = (nu >= nu87_ref - 470) & (nu <= nu87_ref + 30)
    popt87, unc87, rms87 = fit_manifold(nu[mask87], diff[mask87], offsets_87, nu87_ref)
    
    # 85Rb
    m85_loc = (t >= 0.00060) & (t <= 0.00075)
    t85 = t[m85_loc][np.argmax(diff[m85_loc])]
    nu85_ref = nu[np.argmin(np.abs(t - t85))]
    mask85 = (nu >= nu85_ref - 220) & (nu <= nu85_ref + 25)
    popt85, unc85, rms85 = fit_manifold(nu[mask85], diff[mask85], offsets_85, nu85_ref)
    
    rows.append({
        'Run': r_id,
        'Power_uW': pwr,
        '87_Cycling_Gamma': popt87[5], # index 5 is F'=3 (offsets: 0:F1, 1:F2, 2:F3, 3:CO12, 4:CO13, 5:CO23) -> wait! let's check index!
        '87_Cycling_unc': unc87[5],
        '87_CO23_Gamma': popt87[8],
        '87_RMS_mV': rms87,
        '85_Cycling_Gamma': popt85[5],
        '85_Cycling_unc': unc85[5],
        '85_CO34_Gamma': popt85[8],
        '85_RMS_mV': rms85,
    })

df_all = pd.DataFrame(rows)
df_all = df_all.sort_values('Power_uW', ascending=False)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(df_all.to_string(index=False))
