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
    
    # Shift alignment using minimize_scalar
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
    return t, nu, diff, res.x

# Calculate exact Casimir offsets for each manifold
offsets_list = []
for m_idx, m in enumerate(MANIFOLDS_CONFIG):
    E_all = main.calc_peak_offsets(m.ABhfs_guess[0], m.ABhfs_guess[1], m.I, m.J, m.Fp_levels)
    offsets = E_all - E_all[m.ref_index]
    offsets_list.append(offsets)

all_results = []
print(f'Starting fitting for {len(runs)} power variation datasets across all 4 manifolds...')

for r_id, pwr in zip(runs, powers):
    t, nu, diff, shift_val = load_and_calibrate_shifted(r_id)
    
    # 1. 87Rb F=2 reference peak (F'=3) around t in [-0.0036, -0.0031]
    m0 = (t >= -0.0036) & (t <= -0.0031)
    t_ref0 = t[m0][np.argmax(diff[m0])]
    nu_ref0 = nu[np.argmin(np.abs(t - t_ref0))]
    
    # 2. 85Rb F=3 reference peak (F'=4) around t in [0.00055, 0.00080]
    m1 = (t >= 0.00055) & (t <= 0.00080)
    t_ref1 = t[m1][np.argmax(diff[m1])]
    nu_ref1 = nu[np.argmin(np.abs(t - t_ref1))]
    
    # 3. 85Rb F=2 dominant peak CO(2,3) at +61.05 MHz rel to F'=1
    m2 = (t >= 0.0095) & (t <= 0.0115)
    t_p2 = t[m2][np.argmax(diff[m2])]
    nu_ref2 = nu[np.argmin(np.abs(t - t_p2))] - 61.05
    
    # 4. 87Rb F=1 dominant peak CO(1,2) at +150.65 MHz rel to F'=0
    m3 = (t >= 0.0180) & (t <= 0.0195)
    t_p3 = t[m3][np.argmax(diff[m3])]
    nu_ref3 = nu[np.argmin(np.abs(t - t_p3))] - 150.65

    manifold_windows = [
        (0, nu_ref0, -470, 30),   # 87Rb F=2
        (1, nu_ref1, -220, 30),   # 85Rb F=3
        (2, nu_ref2, -30, 130),   # 85Rb F=2
        (3, nu_ref3, -30, 270),   # 87Rb F=1
    ]
    
    for m_idx, nu_ref_init, d_low, d_high in manifold_windows:
        m = MANIFOLDS_CONFIG[m_idx]
        offsets = offsets_list[m_idx]
        mask = (nu >= nu_ref_init + d_low) & (nu <= nu_ref_init + d_high)
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
            
        p0 = [nu_ref_init, float(np.percentile(y, 5)), 0.0] + [12.0]*6 + [float(np.max(y))/3.0]*6
        b_low = [nu_ref_init - 15, -50.0, -1.0] + [2.0]*6 + [0.0]*6
        b_high = [nu_ref_init + 15, 50.0, 1.0] + [48.0]*6 + [1000.0]*6
        
        fit = least_squares(model_fn, p0, bounds=(b_low, b_high), ftol=1e-7, xtol=1e-7)
        
        # Calculate uncertainties
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
            # Valid point criteria: amplitude sufficiently above noise floor and width not pinned to bounds
            valid = (ak >= 0.28) and (gk > 2.05) and (gk < 47.5) and (gk_err < 10.0)
            
            all_results.append({
                'run': r_id,
                'power': pwr,
                'manifold_idx': m_idx,
                'manifold_name': m.name,
                'peak_idx': k,
                'peak_name': m.names[k],
                'gamma': gk,
                'gamma_err': gk_err,
                'amp': ak,
                'valid': valid
            })

df_res = pd.DataFrame(all_results)
df_res.to_csv('tests/all_manifolds_linewidths_shifted.csv', index=False)
print('Finished fitting all 22 runs. Saved data to tests/all_manifolds_linewidths_shifted.csv')

# ==============================================================================
# PLOTTING: 4 Separate Manifold Figures & 1 Master 2x2 Grid Figure
# ==============================================================================
colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', 'p']

# 1. Master 2x2 Figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
axes = axes.flatten()

for m_idx in range(4):
    ax = axes[m_idx]
    m = MANIFOLDS_CONFIG[m_idx]
    sub_df = df_res[df_res['manifold_idx'] == m_idx]
    
    for k in range(6):
        pk_name = m.names[k]
        pk_data = sub_df[sub_df['peak_idx'] == k].sort_values('power')
        v_data = pk_data[pk_data['valid']]
        
        if len(v_data) >= 3:
            ax.errorbar(v_data['power'], v_data['gamma'], yerr=v_data['gamma_err'],
                        fmt=markers[k] + '-', color=colors[k], lw=1.6, ms=5.5, capsize=3,
                        alpha=0.85, label=f"{pk_name}")
            
    ax.axhline(6.07, color='black', ls='--', lw=1.2, alpha=0.75, label='Natural Width (6.07 MHz)')
    ax.set_title(f'({chr(97+m_idx)}) Manifold: {m.name} — Linewidth vs. Pump Power', fontsize=12, fontweight='bold')
    ax.set_xlabel('Pump Power ($\\mu$W)', fontsize=11)
    ax.set_ylabel('FWHM Linewidth $\\Gamma$ (MHz)', fontsize=11)
    ax.set_xlim(-20, 900)
    ax.set_ylim(0, 45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5, ncol=2)

master_path = os.path.join(out_dir, 'session10_power_trends_all_4_manifolds.png')
plt.savefig(master_path, dpi=160)
plt.close()
print(f'Saved master figure to {master_path}')

# 2. Individual High-Resolution Figures for each Manifold
filenames = [
    'session10_power_trends_manifold_87rb_f2.png',
    'session10_power_trends_manifold_85rb_f3.png',
    'session10_power_trends_manifold_85rb_f2.png',
    'session10_power_trends_manifold_87rb_f1.png'
]

for m_idx in range(4):
    m = MANIFOLDS_CONFIG[m_idx]
    sub_df = df_res[df_res['manifold_idx'] == m_idx]
    
    fig_ind, ax_ind = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    
    for k in range(6):
        pk_name = m.names[k]
        pk_data = sub_df[sub_df['peak_idx'] == k].sort_values('power')
        v_data = pk_data[pk_data['valid']]
        
        if len(v_data) >= 3:
            ax_ind.errorbar(v_data['power'], v_data['gamma'], yerr=v_data['gamma_err'],
                            fmt=markers[k] + '-', color=colors[k], lw=1.8, ms=6.5, capsize=3.5,
                            alpha=0.85, label=f"{pk_name}")
            
    ax_ind.axhline(6.07, color='black', ls='--', lw=1.4, alpha=0.8, label='Natural Linewidth $\\Gamma_{\\text{nat}} = 6.07$ MHz')
    ax_ind.set_title(f'Manifold {m.name}: Sub-Doppler Linewidths vs. Pump Power (All 6 Transitions)', fontsize=13, fontweight='bold')
    ax_ind.set_xlabel('Pump Power ($\\mu$W)', fontsize=12)
    ax_ind.set_ylabel('FWHM Linewidth $\\Gamma$ (MHz)', fontsize=12)
    ax_ind.set_xlim(-20, 900)
    ax_ind.set_ylim(0, 45)
    ax_ind.grid(True, alpha=0.3)
    ax_ind.legend(loc='upper right', fontsize=10, ncol=2)
    
    ind_path = os.path.join(out_dir, filenames[m_idx])
    plt.savefig(ind_path, dpi=160)
    plt.close()
    print(f'Saved individual figure {m.name} to {ind_path}')

print('All 5 figures generated successfully!')
