import sys
import os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import least_squares, minimize_scalar
from scipy.interpolate import interp1d
import main
from util import MANIFOLDS_CONFIG

fsr = 374.74
out_dir = 'C:/Users/ilyag/.gemini/antigravity/brain/a9066794-3e7c-4948-9556-40d00e7a0260'

df_pwr = pd.read_excel('data/Session 10/Power variation/Powers.xlsx')
powers = df_pwr['Power_uW'].values
runs = np.arange(1, len(powers) + 1)

def load_and_calibrate_shifted(r_id):
    df_p = pd.read_csv(f'data/Session 10/Power variation/P{r_id}.CSV')
    df_u = pd.read_csv(f'data/Session 10/Power variation/U{r_id}.CSV')
    f_path = f'data/Session 10/Power variation/F{r_id}.CSV'
    if not os.path.exists(f_path): f_path = 'data/Session 10/Power variation/F14.CSV'
    df_f = pd.read_csv(f_path)
    
    t = df_p.iloc[:, 0].values
    p = df_p.iloc[:, 1].values
    u = df_u.iloc[:, 1].values
    index = np.arange(len(p))
    
    # Shift alignment
    u_interp = interp1d(index, u, kind='cubic', bounds_error=False, fill_value='extrapolate')
    pad = 2000
    idx_eval = index[pad:-pad]
    p_eval = p[pad:-pad]
    bg_mask = np.abs(p_eval - u[pad:-pad]) < 0.02
    res = minimize_scalar(lambda s: np.mean((p_eval[bg_mask] - u_interp(idx_eval[bg_mask] + s)) ** 2), bounds=(-300, 300), method='bounded')
    u_shifted = u_interp(index + res.x)
    diff = (p - u_shifted) * 1e3 # in mV
    
    peaks_fp, _ = signal.find_peaks(df_f.iloc[:, 1].values, height=-0.045, prominence=0.005, distance=1000)
    poly = np.poly1d(np.polyfit(t[peaks_fp], np.arange(len(peaks_fp)), 4))
    nu = poly(t) * fsr
    return nu, diff

configs_fit = [
    (0, '87Rb F=2', 450, 970, 920.6),
    (1, '85Rb F=3', 1750, 2150, 2045.7),
    (2, '85Rb F=2', 4750, 5080, 4866.0),
    (3, '87Rb F=1', 7200, 7550, 7256.0),
]

# Calculate exact Casimir offsets for each manifold
manifold_offsets = []
for m_idx, name, w0, w1, ref_g in configs_fit:
    m = MANIFOLDS_CONFIG[m_idx]
    E_all = main.calc_peak_offsets(m.ABhfs_guess[0], m.ABhfs_guess[1], m.I, m.J, m.Fp_levels)
    offsets = E_all - E_all[m.ref_index]
    manifold_offsets.append(offsets)

all_results = []

print('Processing all 22 runs with shift alignment across all 4 manifolds...')
for r_id, pwr in zip(runs, powers):
    nu, diff = load_and_calibrate_shifted(r_id)
    
    for m_idx, name, w0, w1, ref_guess in configs_fit:
        m = MANIFOLDS_CONFIG[m_idx]
        offsets = manifold_offsets[m_idx]
        mask = (nu >= w0) & (nu <= w1)
        x = nu[mask]
        y = diff[mask]
        
        def model_fn(p):
            nu_ref, c0, c1 = p[:3]
            gammas = p[3:9]
            amps = p[9:15]
            y_out = c0 + c1 * (x - nu_ref)
            for k in range(6):
                y_out += amps[k] / (1.0 + 4.0 * ((x - (nu_ref + offsets[k])) / gammas[k])**2)
            return y_out - y
            
        p0 = [ref_guess, float(np.percentile(y, 5)), 0.0] + [15.0]*6 + [float(np.max(y))/3.0]*6
        b_low = [ref_guess - 25, -50.0, -1.0] + [2.0]*6 + [0.0]*6
        b_high = [ref_guess + 25, 50.0, 1.0] + [50.0]*6 + [1000.0]*6
        
        fit = least_squares(model_fn, p0, bounds=(b_low, b_high), ftol=1e-7, xtol=1e-7)
        
        # Uncertainties
        dof = len(y) - len(p0)
        s_sq = 2.0 * fit.cost / max(1, dof)
        try:
            cov = np.linalg.inv(fit.jac.T @ fit.jac) * s_sq
            unc = np.sqrt(np.diagonal(cov))
        except:
            unc = np.zeros_like(fit.x)
            
        for k in range(6):
            gk = fit.x[3+k]
            gk_err = unc[3+k]
            ak = fit.x[9+k]
            all_results.append({
                'run': r_id,
                'power': pwr,
                'manifold_idx': m_idx,
                'manifold_name': name,
                'peak_idx': k,
                'peak_name': m.names[k],
                'gamma': gk,
                'gamma_err': gk_err,
                'amp': ak,
                'valid': (ak >= 0.25) and (gk > 2.05) and (gk < 49.5) and (gk_err < 15.0)
            })

df_res = pd.DataFrame(all_results)
df_res.to_csv('tests/all_manifolds_linewidths.csv', index=False)
print('Saved all fit results to tests/all_manifolds_linewidths.csv')

# ==============================================================================
# PLOTTING: 4 Separate Manifold Plots in a 2x2 Master Figure
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
axes = axes.flatten()

# Harmonized distinct color palette and markers for the 6 peaks
colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
markers = ['o', 's', '^', 'D', 'v', 'p']

for m_idx, (name, ax) in enumerate(zip([c[1] for c in configs_fit], axes)):
    sub_df = df_res[df_res['manifold_idx'] == m_idx]
    m = MANIFOLDS_CONFIG[m_idx]
    
    for k in range(6):
        pk_name = m.names[k]
        pk_data = sub_df[sub_df['peak_idx'] == k].sort_values('power')
        
        # Valid points only (above noise threshold)
        v_data = pk_data[pk_data['valid']]
        
        if len(v_data) > 0:
            ax.errorbar(v_data['power'], v_data['gamma'], yerr=v_data['gamma_err'],
                        fmt=markers[k] + '-', color=colors[k], lw=1.6, ms=5.5, capsize=3,
                        label=f"{pk_name}")
            
    # Reference natural linewidth (6.07 MHz)
    ax.axhline(6.07, color='black', ls='--', lw=1.2, alpha=0.75, label='Natural Linewidth (6.07 MHz)')
    
    ax.set_title(f'({chr(97+m_idx)}) Manifold: {name} — Linewidth vs. Pump Power', fontsize=12, fontweight='bold')
    ax.set_xlabel('Pump Power ($\\mu$W)', fontsize=11)
    ax.set_ylabel('FWHM Linewidth $\\Gamma$ (MHz)', fontsize=11)
    ax.set_xlim(-20, 900)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5, ncol=2)

fig_path = os.path.join(out_dir, 'session10_all_manifolds_power_trends.png')
plt.savefig(fig_path, dpi=160)
plt.close()
print(f'Saved 4-manifold power trends figure to {fig_path}')
