import os
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.optimize import least_squares
import main

fsr = 374.74

# Literature values for Casimir formula:
# 87Rb 5P_3/2: A = 84.7185 MHz, B = 12.4965 MHz
# 85Rb 5P_3/2: A = 25.0020 MHz, B = 25.7900 MHz
E87_all = main.calc_peak_offsets(84.7185, 12.4965, 1.5, 1.5, [1, 2, 3])
offsets_87 = E87_all - E87_all[2] # rel to F'=3

E85_all = main.calc_peak_offsets(25.0020, 25.7900, 2.5, 1.5, [2, 3, 4])
offsets_85 = E85_all - E85_all[2] # rel to F'=4

names_87 = ["F'=1", "F'=2", "F'=3 (Cycling)", "CO(1,2)", "CO(1,3)", "CO(2,3)"]
names_85 = ["F'=2", "F'=3", "F'=4 (Cycling)", "CO(2,3)", "CO(2,4)", "CO(3,4)"]

def load_and_calibrate(r_id):
    df_p = pd.read_csv(f'data/Session 10/Power variation/P{r_id}.CSV')
    df_u = pd.read_csv(f'data/Session 10/Power variation/U{r_id}.CSV')
    
    # Check if F file exists, else use F14 or F16 for Run 15
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

def fit_manifold_model(nu_man, diff_man, offsets, ref_guess, pnames):
    # Model parameters: [nu_ref, c0, c1, gamma_0..5, amp_0..5] (15 params)
    def model_fn(p, x):
        nu_ref, c0, c1 = p[:3]
        gammas = p[3:9]
        amps = p[9:15]
        y = c0 + c1 * (x - nu_ref)
        for k in range(6):
            y += amps[k] / (1.0 + 4.0 * ((x - (nu_ref + offsets[k])) / gammas[k])**2)
        return y

    def res(p):
        return model_fn(p, nu_man) - diff_man

    # Initial guesses
    max_idx = np.argmax(diff_man)
    c0_init = float(np.percentile(diff_man, 5))
    amp_init = float(np.max(diff_man) - c0_init)
    
    p0 = [ref_guess, c0_init, 0.0] + [8.0]*6 + [amp_init/3.0]*6
    # bounds: nu_ref +/- 30 MHz, c0 +/- 10 mV, c1 +/- 1 mV/MHz, gammas in [2.0, 50.0] MHz, amps >= 0
    b_low = [ref_guess - 40, -0.05, -0.001] + [2.0]*6 + [0.0]*6
    b_high = [ref_guess + 40, 0.05, 0.001] + [40.0]*6 + [1.0]*6
    
    fit = least_squares(res, p0, bounds=(b_low, b_high), ftol=1e-7, xtol=1e-7)
    
    # Calculate uncertainties
    dof = len(diff_man) - len(p0)
    s_sq = 2.0 * fit.cost / max(1, dof)
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * s_sq
        unc = np.sqrt(np.diagonal(cov))
    except:
        unc = np.zeros_like(fit.x)
        
    y_fit = model_fn(fit.x, nu_man)
    residuals = diff_man - y_fit
    res_rms = np.std(residuals) * 1e3 # in mV
    
    return fit.x, unc, residuals, res_rms, y_fit

print('Running fits on selected runs across the power series...')
df_pwr = pd.read_excel('data/Session 10/Power variation/Powers.xlsx')
test_runs = [1, 5, 7, 11, 14, 17, 21, 22]

for r_id in test_runs:
    pwr = df_pwr['Power_uW'].iloc[r_id-1]
    t, nu, diff = load_and_calibrate(r_id)
    
    # 87Rb Manifold: nu range ~ [450, 960] MHz
    # Cycling peak is around t = -0.0033
    m87_loc = (t >= -0.0035) & (t <= -0.0031)
    t87 = t[m87_loc][np.argmax(diff[m87_loc])]
    nu87_ref = poly_val = nu[np.argmin(np.abs(t - t87))]
    
    # Manifold window: from -460 MHz to +30 MHz relative to cycling peak
    mask_87 = (nu >= nu87_ref - 470) & (nu <= nu87_ref + 30)
    x_87 = nu[mask_87]
    y_87 = diff[mask_87]
    
    p_opt, p_unc, res_87, rms_87, y_fit_87 = fit_manifold_model(x_87, y_87, offsets_87, nu87_ref, names_87)
    
    print(f'\n=== Run {r_id} ({pwr:.1f} uW) -- 87Rb F=2 Manifold Multiplet Fit ===')
    print(f'Residual RMS: {rms_87:.3f} mV')
    print(f'Baseline c0 = {p_opt[1]*1e3:.2f} mV, c1 = {p_opt[2]*1e3:.4f} mV/MHz')
    for k in range(6):
        gk = p_opt[3+k]
        gk_err = p_unc[3+k]
        ak = p_opt[9+k] * 1e3
        print(f'  {names_87[k]:18s}: FWHM = {gk:5.2f} +/- {gk_err:4.2f} MHz | Amp = {ak:6.2f} mV | Offset = {offsets_87[k]:6.1f} MHz')
