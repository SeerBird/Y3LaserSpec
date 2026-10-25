import os
import sys
sys.path.insert(0, '.')
import main
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks

PEAK_NAMES = {
    '87Rb F=2': ["F'=1", "F'=2", "F'=3", "CO(1,2)", "CO(1,3)", "CO(2,3)"],
    '85Rb F=3': ["F'=2", "F'=3", "F'=4", "CO(2,3)", "CO(2,4)", "CO(3,4)"],
    '85Rb F=2': ["F'=1", "F'=2", "F'=3", "CO(1,2)", "CO(1,3)", "CO(2,3)"],
    '87Rb F=1': ["F'=0", "F'=1", "F'=2", "CO(0,1)", "CO(0,2)", "CO(1,2)"],
}

configs = [
    ('87Rb F=2', 2, 0.5, 1.5, 1.5, [1, 2, 3], 2, (84.72, 12.50)),
    ('85Rb F=3', 3, 0.5, 1.5, 2.5, [2, 3, 4], 2, (25.00, 25.80)),
    ('85Rb F=2', 2, 0.5, 1.5, 2.5, [1, 2, 3], 0, (25.00, 25.80)),
    ('87Rb F=1', 1, 0.5, 1.5, 1.5, [0, 1, 2], 0, (84.72, 12.50)),
]

def prepare_slices(fp, p, u, windows, fsr=374.74):
    freq, _, pumped, unpumped = main.prep_data(fp, p, u, align_shift=True)
    diff = pumped - unpumped
    slices = []
    for cfg, (w0, w1) in zip(configs, windows):
        mask = (freq >= w0) & (freq <= w1)
        x = freq[mask] * fsr
        y = diff[mask]
        ref_idx = cfg[6]
        A_init, B_init = cfg[7]
        E_init = main.calc_peak_offsets(A_init, B_init, cfg[4], cfg[3], cfg[5])
        offsets_init = E_init - E_init[ref_idx]
        dpeaks, _ = find_peaks(y, prominence=0.005)
        if len(dpeaks) > 0:
            if cfg[0] == '85Rb F=2':
                nu_ref_init = float(x[dpeaks[-1]] - offsets_init[5])
            elif cfg[0] == '87Rb F=1':
                nu_ref_init = float(x[dpeaks[-1]] - offsets_init[2])
            else:
                k_max = np.argmax(offsets_init)
                nu_ref_init = float(x[dpeaks[-1]] - offsets_init[k_max])
        else:
            nu_ref_init = float(0.5 * (x[0] + x[-1]))
        bg_init = float(np.percentile(y, 10))
        amp_init = float(np.max(y) - bg_init) / 3.0
        slices.append({
            'name': cfg[0], 'cfg': cfg, 'x': x, 'y': y,
            'nu_ref_init': nu_ref_init, 'bg_init': bg_init,
            'amp_init': amp_init, 'ref_idx': ref_idx
        })
    return slices

def fit_1gamma(d):
    cfg, ref_idx, x, y = d['cfg'], d['ref_idx'], d['x'], d['y']
    def res(p):
        nu_ref, A, B, gamma, bg = p[:5]
        amps = p[5:]
        E_all = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
        offsets = E_all - E_all[ref_idx]
        y_eval = np.full_like(x, bg)
        for k in range(6):
            y_eval += amps[k] / (1.0 + 4.0 * ((x - (nu_ref + offsets[k])) / gamma)**2)
        return y_eval - y
    p0 = [d['nu_ref_init'], cfg[7][0], cfg[7][1], 20.0, d['bg_init']] + [d['amp_init']]*6
    fit = least_squares(res, p0, bounds=([x[0]-50, 5, 0, 8.0, -1.0] + [0.0]*6,
                                         [x[-1]+50, 150, 60, 35.0, 1.0] + [5.0]*6))
    def eval_fn(x_in):
        nu_ref, A, B, gamma, bg = fit.x[:5]
        amps = fit.x[5:]
        E_all = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
        offsets = E_all - E_all[ref_idx]
        y_eval = np.full_like(x_in, bg)
        for k in range(6):
            y_eval += amps[k] / (1.0 + 4.0 * ((x_in - (nu_ref + offsets[k])) / gamma)**2)
        return y_eval
    return fit, eval_fn

def fit_6gamma(d):
    cfg, ref_idx, x, y = d['cfg'], d['ref_idx'], d['x'], d['y']
    def res(p):
        nu_ref, A, B, bg = p[:4]
        gammas = p[4:10]
        amps = p[10:16]
        E_all = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
        offsets = E_all - E_all[ref_idx]
        y_eval = np.full_like(x, bg)
        for k in range(6):
            y_eval += amps[k] / (1.0 + 4.0 * ((x - (nu_ref + offsets[k])) / gammas[k])**2)
        return y_eval - y
    p0 = [d['nu_ref_init'], cfg[7][0], cfg[7][1], d['bg_init']] + [20.0]*6 + [d['amp_init']]*6
    fit = least_squares(res, p0, bounds=([x[0]-50, 5.0, 0.0, -1.0] + [5.0]*6 + [0.0]*6,
                                         [x[-1]+50, 150.0, 60.0, 1.0] + [45.0]*6 + [5.0]*6))
    def eval_fn(x_in):
        nu_ref, A, B, bg = fit.x[:4]
        gammas = fit.x[4:10]
        amps = fit.x[10:16]
        E_all = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
        offsets = E_all - E_all[ref_idx]
        y_eval = np.full_like(x_in, bg)
        comps = []
        for k in range(6):
            comp = amps[k] / (1.0 + 4.0 * ((x_in - (nu_ref + offsets[k])) / gammas[k])**2)
            y_eval += comp
            comps.append(comp)
        return y_eval, bg, comps, nu_ref + offsets, gammas
    return fit, eval_fn

os.makedirs('figures/free_linewidths', exist_ok=True)
windows1 = [(3.7, 5.2), (7.4, 8.1), (15.4, 15.9), (21.8, 22.7)]
slices1 = prepare_slices('data/Session 9/F1.CSV', 'data/Session 9/P1.CSV', 'data/Session 9/U1.CSV', windows1)

fig, axes = plt.subplots(4, 2, figsize=(15, 11), gridspec_kw={'width_ratios': [2.2, 1.0], 'hspace': 0.38, 'wspace': 0.22})
fig.suptitle("Protocol 1: 1 Uniform Linewidth vs. 6 Free Independent Linewidths per Manifold", fontsize=12, fontweight='bold')

for m_idx, d in enumerate(slices1):
    ax_fit, ax_res = axes[m_idx, 0], axes[m_idx, 1]
    x, y = d['x'], d['y']
    x_dense = np.linspace(x[0], x[-1], 600)
    
    fit_1g, eval_1g = fit_1gamma(d)
    fit_6g, eval_6g = fit_6gamma(d)
    
    y_1g_dense = eval_1g(x_dense)
    y_1g_pts = eval_1g(x)
    res_1g = y_1g_pts - y
    
    y_6g_dense, bg_6g, comps, peak_locs, gammas = eval_6g(x_dense)
    y_6g_pts, _, _, _, _ = eval_6g(x)
    res_6g = y_6g_pts - y
    
    ax_fit.plot(x, y, 'k.', markersize=2.5, alpha=0.40, label='Data (P - U)')
    ax_fit.plot(x_dense, y_1g_dense, '--', color='#1f77b4', lw=1.5,
                label=f"1-Gamma ($\\Gamma={fit_1g.x[3]:.1f}\\text{{ MHz}}$, A={fit_1g.x[1]:.1f}, Cost={fit_1g.cost:.3f})")
    ax_fit.plot(x_dense, y_6g_dense, '-', color='#d62728', lw=1.8,
                label=f"6-Gamma (Free widths, A={fit_6g.x[1]:.1f}, Cost={fit_6g.cost:.3f})")
                
    pnames = PEAK_NAMES[d['name']]
    y_max = np.max(y)
    for k_pk in range(6):
        c_line = 'navy' if k_pk < 3 else 'darkgreen'
        y_txt = y_max * (0.90 if k_pk < 3 else 0.72)
        ax_fit.axvline(peak_locs[k_pk], color=c_line, linestyle=':', alpha=0.45, lw=0.9)
        ax_fit.text(peak_locs[k_pk], y_txt, f"{pnames[k_pk]}\n$\\Gamma={gammas[k_pk]:.1f}$", color=c_line, fontsize=6.8,
                    ha='center', rotation=90, bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

    ax_fit.set_title(f"Manifold {m_idx+1}: {d['name']} | $\\Gamma_k \\in [{np.min(gammas):.1f}, {np.max(gammas):.1f}]\\text{{ MHz}}$",
                     fontweight='bold', fontsize=10, loc='left')
    ax_fit.set_ylabel("Diff Abs (V)")
    ax_fit.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax_fit.grid(True, linestyle=':', alpha=0.6)

    ax_res.plot(x, res_1g, color='#1f77b4', lw=0.9, alpha=0.8, label=f"1-Gamma (RMS: {np.std(res_1g)*1e3:.1f} mV)")
    ax_res.plot(x, res_6g, color='#d62728', lw=1.0, alpha=0.8, label=f"6-Gamma (RMS: {np.std(res_6g)*1e3:.1f} mV)")
    ax_res.axhline(0, color='gray', linestyle='--', lw=0.8)
    ax_res.set_ylabel("Residual (V)")
    ax_res.set_title(f"Residuals ({d['name']})", fontsize=10, loc='left')
    ax_res.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax_res.grid(True, linestyle=':', alpha=0.6)

    if m_idx == 3:
        ax_fit.set_xlabel("Frequency relative to FSR (MHz)")
        ax_res.set_xlabel("Frequency (MHz)")

out_path = 'figures/free_linewidths/prot1_1gamma_vs_6gamma.png'
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Saved: {out_path}")
