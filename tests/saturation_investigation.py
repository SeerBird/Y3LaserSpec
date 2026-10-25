import os
import sys
import shutil
sys.path.insert(0, '.')
import main
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks

# Configure matplotlib style
plt.style.use('default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8.5
plt.rcParams['figure.titlesize'] = 12

COUPLINGS = {
    '87Rb F=2': np.array([0.05714286, 0.28571429, 0.8,        0.07142857, 0.2,        1.0]),
    '85Rb F=3': np.array([0.06349206, 0.22222222, 0.51428571, 0.12345679, 0.28571429, 1.0]),
    '85Rb F=2': np.array([0.49586777, 0.64285714, 0.51428571, 0.96428571, 0.77142857, 1.0]),
    '87Rb F=1': np.array([0.32,       0.8,        0.8,        0.4,        0.4,        1.0]),
}

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

def prepare_manifold_slices(fp_file, pump_file, unpump_file, windows, fsr=374.74):
    freq, _, p, u = main.prep_data(fp_file, pump_file, unpump_file, align_shift=True)
    diff = p - u
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
            'C': COUPLINGS[cfg[0]], 'nu_ref_init': nu_ref_init,
            'bg_init': bg_init, 'amp_init': amp_init, 'ref_idx': ref_idx
        })
    return slices

def run_all_fits(slices):
    results = {}
    
    # Model 0: Single uniform gamma per manifold
    results['M0'] = []
    for d in slices:
        cfg = d['cfg']
        ref_idx = d['ref_idx']
        x, y = d['x'], d['y']
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
        b_low = [x[0]-50, 5, 0, 8.0, -1.0] + [0.0]*6
        b_high = [x[-1]+50, 150, 60, 32.0, 1.0] + [5.0]*6
        fit = least_squares(res, p0, bounds=(b_low, b_high), ftol=1e-6, xtol=1e-6)
        
        # Function to evaluate
        def eval_fn(x_in, p=fit.x):
            nu_ref, A, B, gamma, bg = p[:5]
            amps = p[5:]
            E_all = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
            offsets = E_all - E_all[ref_idx]
            y_eval = np.full_like(x_in, bg)
            for k in range(6):
                y_eval += amps[k] / (1.0 + 4.0 * ((x_in - (nu_ref + offsets[k])) / gamma)**2)
            return y_eval
            
        results['M0'].append({'fit': fit, 'eval': eval_fn, 'gamma': fit.x[3], 'A': fit.x[1], 'B': fit.x[2]})
        
    # Model 2: Shared isotope
    d0, d3 = slices[0], slices[3]
    def res_iso87(p):
        nu0, A0, B0, bg0, nu3, A3, B3, bg3, g0, s = p[:10]
        a0 = p[10:16]
        a3 = p[16:22]
        E0 = main.calc_peak_offsets(A0, B0, d0['cfg'][4], d0['cfg'][3], d0['cfg'][5])
        off0 = E0 - E0[d0['ref_idx']]
        g0_v = g0 * np.sqrt(1.0 + s * d0['C'])
        y0 = np.full_like(d0['x'], bg0)
        for k in range(6):
            y0 += (a0[k] * d0['C'][k]) / (1.0 + 4.0 * ((d0['x'] - (nu0 + off0[k])) / g0_v[k])**2)
            
        E3 = main.calc_peak_offsets(A3, B3, d3['cfg'][4], d3['cfg'][3], d3['cfg'][5])
        off3 = E3 - E3[d3['ref_idx']]
        g3_v = g0 * np.sqrt(1.0 + s * d3['C'])
        y3 = np.full_like(d3['x'], bg3)
        for k in range(6):
            y3 += (a3[k] * d3['C'][k]) / (1.0 + 4.0 * ((d3['x'] - (nu3 + off3[k])) / g3_v[k])**2)
        return np.concatenate([y0 - d0['y'], y3 - d3['y']])
        
    p0_87 = [d0['nu_ref_init'], 84.72, 12.50, d0['bg_init'], d3['nu_ref_init'], 84.72, 12.50, d3['bg_init'], 18.0, 0.5] + [d0['amp_init']]*6 + [d3['amp_init']]*6
    b_low_87 = [d0['x'][0]-50, 5, 0, -1, d3['x'][0]-50, 5, 0, -1, 5.0, 0.0] + [0.0]*12
    b_high_87 = [d0['x'][-1]+50, 150, 60, 1, d3['x'][-1]+50, 150, 60, 1, 35.0, 20.0] + [5.0]*12
    fit87 = least_squares(res_iso87, p0_87, bounds=(b_low_87, b_high_87), ftol=1e-6, xtol=1e-6)

    d1, d2 = slices[1], slices[2]
    def res_iso85(p):
        nu1, A1, B1, bg1, nu2, A2, B2, bg2, g0, s = p[:10]
        a1 = p[10:16]
        a2 = p[16:22]
        E1 = main.calc_peak_offsets(A1, B1, d1['cfg'][4], d1['cfg'][3], d1['cfg'][5])
        off1 = E1 - E1[d1['ref_idx']]
        g1_v = g0 * np.sqrt(1.0 + s * d1['C'])
        y1 = np.full_like(d1['x'], bg1)
        for k in range(6):
            y1 += (a1[k] * d1['C'][k]) / (1.0 + 4.0 * ((d1['x'] - (nu1 + off1[k])) / g1_v[k])**2)
            
        E2 = main.calc_peak_offsets(A2, B2, d2['cfg'][4], d2['cfg'][3], d2['cfg'][5])
        off2 = E2 - E2[d2['ref_idx']]
        g2_v = g0 * np.sqrt(1.0 + s * d2['C'])
        y2 = np.full_like(d2['x'], bg2)
        for k in range(6):
            y2 += (a2[k] * d2['C'][k]) / (1.0 + 4.0 * ((d2['x'] - (nu2 + off2[k])) / g2_v[k])**2)
        return np.concatenate([y1 - d1['y'], y2 - d2['y']])
        
    p0_85 = [d1['nu_ref_init'], 25.0, 25.8, d1['bg_init'], d2['nu_ref_init'], 25.0, 25.8, d2['bg_init'], 18.0, 0.5] + [d1['amp_init']]*6 + [d2['amp_init']]*6
    b_low_85 = [d1['x'][0]-50, 5, 0, -1, d2['x'][0]-50, 5, 0, -1, 5.0, 0.0] + [0.0]*12
    b_high_85 = [d1['x'][-1]+50, 150, 60, 1, d2['x'][-1]+50, 150, 60, 1, 35.0, 20.0] + [5.0]*12
    fit85 = least_squares(res_iso85, p0_85, bounds=(b_low_85, b_high_85), ftol=1e-6, xtol=1e-6)

    def eval_iso(m_idx, x_in):
        if m_idx == 0:
            nu0, A0, B0, bg0 = fit87.x[:4]
            g0, s = fit87.x[8], fit87.x[9]
            amps = fit87.x[10:16]
            cfg, C, ref = d0['cfg'], d0['C'], d0['ref_idx']
            E = main.calc_peak_offsets(A0, B0, cfg[4], cfg[3], cfg[5])
            off = E - E[ref]
            gammas = g0 * np.sqrt(1.0 + s * C)
            y_out = np.full_like(x_in, bg0)
            for k in range(6):
                y_out += (amps[k] * C[k]) / (1.0 + 4.0 * ((x_in - (nu0 + off[k])) / gammas[k])**2)
            return y_out
        elif m_idx == 3:
            nu3, A3, B3, bg3 = fit87.x[4:8]
            g0, s = fit87.x[8], fit87.x[9]
            amps = fit87.x[16:22]
            cfg, C, ref = d3['cfg'], d3['C'], d3['ref_idx']
            E = main.calc_peak_offsets(A3, B3, cfg[4], cfg[3], cfg[5])
            off = E - E[ref]
            gammas = g0 * np.sqrt(1.0 + s * C)
            y_out = np.full_like(x_in, bg3)
            for k in range(6):
                y_out += (amps[k] * C[k]) / (1.0 + 4.0 * ((x_in - (nu3 + off[k])) / gammas[k])**2)
            return y_out
        elif m_idx == 1:
            nu1, A1, B1, bg1 = fit85.x[:4]
            g0, s = fit85.x[8], fit85.x[9]
            amps = fit85.x[10:16]
            cfg, C, ref = d1['cfg'], d1['C'], d1['ref_idx']
            E = main.calc_peak_offsets(A1, B1, cfg[4], cfg[3], cfg[5])
            off = E - E[ref]
            gammas = g0 * np.sqrt(1.0 + s * C)
            y_out = np.full_like(x_in, bg1)
            for k in range(6):
                y_out += (amps[k] * C[k]) / (1.0 + 4.0 * ((x_in - (nu1 + off[k])) / gammas[k])**2)
            return y_out
        else: # m_idx == 2
            nu2, A2, B2, bg2 = fit85.x[4:8]
            g0, s = fit85.x[8], fit85.x[9]
            amps = fit85.x[16:22]
            cfg, C, ref = d2['cfg'], d2['C'], d2['ref_idx']
            E = main.calc_peak_offsets(A2, B2, cfg[4], cfg[3], cfg[5])
            off = E - E[ref]
            gammas = g0 * np.sqrt(1.0 + s * C)
            y_out = np.full_like(x_in, bg2)
            for k in range(6):
                y_out += (amps[k] * C[k]) / (1.0 + 4.0 * ((x_in - (nu2 + off[k])) / gammas[k])**2)
            return y_out

    results['M2_eval'] = eval_iso
    results['M2_fit87'] = fit87
    results['M2_fit85'] = fit85
    
    # Model 3: Global s_beam & Gamma_0
    def res_global(p):
        residuals = []
        g0, s = p[16], p[17]
        for m_idx in range(4):
            d = slices[m_idx]
            cfg = d['cfg']
            nu_ref, A, B, bg = p[m_idx*4 : m_idx*4 + 4]
            amps = p[18 + m_idx*6 : 18 + (m_idx+1)*6]
            E = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
            offsets = E - E[d['ref_idx']]
            gammas = g0 * np.sqrt(1.0 + s * d['C'])
            y_eval = np.full_like(d['x'], bg)
            for k in range(6):
                y_eval += (amps[k] * d['C'][k]) / (1.0 + 4.0 * ((d['x'] - (nu_ref + offsets[k])) / gammas[k])**2)
            residuals.append(y_eval - d['y'])
        return np.concatenate(residuals)

    p0_glob = []
    b_low_glob = []
    b_high_glob = []
    for d in slices:
        p0_glob.extend([d['nu_ref_init'], d['cfg'][7][0], d['cfg'][7][1], d['bg_init']])
        b_low_glob.extend([d['x'][0]-50, 5.0, 0.0, -1.0])
        b_high_glob.extend([d['x'][-1]+50, 150.0, 60.0, 1.0])
    p0_glob.extend([18.0, 0.5])
    b_low_glob.extend([5.0, 0.0])
    b_high_glob.extend([35.0, 20.0])
    for d in slices:
        p0_glob.extend([d['amp_init']]*6)
        b_low_glob.extend([0.0]*6)
        b_high_glob.extend([5.0]*6)

    fit_glob = least_squares(res_global, p0_glob, bounds=(b_low_glob, b_high_glob), ftol=1e-6, xtol=1e-6)

    def eval_glob(m_idx, x_in):
        d = slices[m_idx]
        cfg = d['cfg']
        nu_ref, A, B, bg = fit_glob.x[m_idx*4 : m_idx*4 + 4]
        g0, s = fit_glob.x[16], fit_glob.x[17]
        amps = fit_glob.x[18 + m_idx*6 : 18 + (m_idx+1)*6]
        E = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
        offsets = E - E[d['ref_idx']]
        gammas = g0 * np.sqrt(1.0 + s * d['C'])
        y_eval = np.full_like(x_in, bg)
        components = []
        for k in range(6):
            comp = (amps[k] * d['C'][k]) / (1.0 + 4.0 * ((x_in - (nu_ref + offsets[k])) / gammas[k])**2)
            y_eval += comp
            components.append(comp)
        return y_eval, bg, components, nu_ref + offsets, gammas

    results['M3_eval'] = eval_glob
    results['M3_fit'] = fit_glob
    
    return results

def make_protocol_plot(prot_num, slices, fits, out_path):
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), gridspec_kw={'width_ratios': [2.2, 1.0], 'hspace': 0.35, 'wspace': 0.20})
    
    m3_fit = fits['M3_fit']
    g0_glob = m3_fit.x[16]
    s_glob = m3_fit.x[17]
    
    fig.suptitle(f"Protocol {prot_num}: Saturated Absorption Hyperfine Fits Comparison\n"
                 f"Global Model: $\\Gamma_0 = {g0_glob:.2f}\\text{{ MHz}}$, $s = {s_glob:.3f}$",
                 fontsize=13, fontweight='bold', y=0.98)
                 
    for m_idx, d in enumerate(slices):
        ax_fit = axes[m_idx, 0]
        ax_res = axes[m_idx, 1]
        
        x = d['x']
        y = d['y']
        
        # Dense x for plotting smooth curves
        x_dense = np.linspace(x[0], x[-1], 600)
        
        # Model 0: Uniform gamma
        m0_data = fits['M0'][m_idx]
        y_m0_dense = m0_data['eval'](x_dense)
        y_m0_pts = m0_data['eval'](x)
        res_m0 = y_m0_pts - y
        
        # Model 2: Shared isotope
        y_m2_dense = fits['M2_eval'](m_idx, x_dense)
        y_m2_pts = fits['M2_eval'](m_idx, x)
        res_m2 = y_m2_pts - y
        
        # Model 3: Global s
        y_m3_dense, bg_dense, comps, peak_locs, gammas = fits['M3_eval'](m_idx, x_dense)
        y_m3_pts, _, _, _, _ = fits['M3_eval'](m_idx, x)
        res_m3 = y_m3_pts - y
        
        # Plot data & fits on left
        ax_fit.plot(x, y, 'k.', markersize=2.5, alpha=0.40, label='Data (P - U)')
        ax_fit.plot(x_dense, y_m0_dense, '--', color='#1f77b4', lw=1.3,
                    label=f"M0 Uniform (4 $\\Gamma$) | $\\Gamma={m0_data['gamma']:.1f}\\text{{ MHz}}$ (A={m0_data['A']:.1f})")
        ax_fit.plot(x_dense, y_m2_dense, '-', color='#2ca02c', lw=1.6,
                    label=f"M2 Per-Isotope (2 $\\Gamma, 2 s$)")
        ax_fit.plot(x_dense, y_m3_dense, ':', color='#d62728', lw=1.8,
                    label=f"M3 Global (1 $\\Gamma, 1 s$) | $\\Gamma_0={g0_glob:.1f}, s={s_glob:.2f}$ (A={fits['M3_fit'].x[m_idx*4 + 1]:.1f})")
        
        # Mark peak positions
        pnames = PEAK_NAMES[d['name']]
        y_max = np.max(y)
        for k_pk in range(6):
            if k_pk < 3:
                ax_fit.axvline(peak_locs[k_pk], color='navy', linestyle=':', alpha=0.55, lw=1)
                ax_fit.text(peak_locs[k_pk], y_max*0.92, pnames[k_pk], color='navy',
                            fontsize=7.5, ha='center', rotation=90, bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))
            else:
                ax_fit.axvline(peak_locs[k_pk], color='darkgreen', linestyle=':', alpha=0.4, lw=0.9)
                ax_fit.text(peak_locs[k_pk], y_max*0.75, pnames[k_pk], color='darkgreen',
                            fontsize=7.0, ha='center', rotation=90, bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))
                            
        ax_fit.set_title(f"Manifold {m_idx+1}: {d['name']} | $\\Gamma_k \\in [{np.min(gammas):.1f}, {np.max(gammas):.1f}]\\text{{ MHz}}$",
                         fontweight='bold', fontsize=10, loc='left')
        ax_fit.set_ylabel("Diff Abs (V)")
        ax_fit.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
        ax_fit.grid(True, linestyle=':', alpha=0.6)
        
        # Plot residuals on right
        ax_res.plot(x, res_m0, color='#1f77b4', lw=0.9, alpha=0.7, label=f"M0 Uniform (RMS: {np.std(res_m0)*1e3:.1f} mV)")
        ax_res.plot(x, res_m2, color='#2ca02c', lw=1.0, alpha=0.8, label=f"M2 Per-Isotope (RMS: {np.std(res_m2)*1e3:.1f} mV)")
        ax_res.plot(x, res_m3, color='#d62728', lw=0.9, linestyle='--', alpha=0.7, label=f"M3 Global (RMS: {np.std(res_m3)*1e3:.1f} mV)")
        ax_res.axhline(0, color='gray', linestyle='--', lw=0.8)
        ax_res.set_ylabel("Residual (V)")
        ax_res.set_title(f"Residuals ({d['name']})", fontsize=10, loc='left')
        ax_res.legend(loc='upper right', framealpha=0.9)
        ax_res.grid(True, linestyle=':', alpha=0.6)
        
        if m_idx == 3:
            ax_fit.set_xlabel("Frequency relative to FSR (MHz)")
            ax_res.set_xlabel("Frequency (MHz)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

def make_component_decomposition_plot(prot_num, slices, fits, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    axes = axes.flatten()
    
    m3_fit = fits['M3_fit']
    g0_glob = m3_fit.x[16]
    s_glob = m3_fit.x[17]
    
    fig.suptitle(f"Protocol {prot_num}: Lorentzian Components Decomposed with Global Power Broadening\n"
                 f"$\\Gamma_0 = {g0_glob:.2f}\\text{{ MHz}}$, $s = {s_glob:.3f}$ ($I / I_{{sat}}$)",
                 fontsize=12, fontweight='bold', y=0.98)
                 
    colors_true = ['#1f77b4', '#2ca02c', '#9467bd']
    colors_co = ['#ff7f0e', '#8c564b', '#e377c2']
    
    for m_idx, d in enumerate(slices):
        ax = axes[m_idx]
        x = d['x']
        y = d['y']
        x_dense = np.linspace(x[0], x[-1], 600)
        
        y_tot, bg, comps, peak_locs, gammas = fits['M3_eval'](m_idx, x_dense)
        pnames = PEAK_NAMES[d['name']]
        
        ax.plot(x, y, 'k.', markersize=2.5, alpha=0.35, label='Experimental Data')
        ax.plot(x_dense, y_tot, 'r-', lw=2.0, label='Total Saturated Fit')
        ax.plot(x_dense, np.full_like(x_dense, bg), 'k--', lw=0.9, alpha=0.7, label='Fitted Baseline')
        
        for k in range(6):
            c = colors_true[k] if k < 3 else colors_co[k-3]
            ls = '-' if k < 3 else '--'
            tag = "True" if k < 3 else "CO"
            lbl = f"{pnames[k]} ($\\Gamma={gammas[k]:.1f}\\text{{ MHz}}$)"
            ax.plot(x_dense, bg + comps[k], linestyle=ls, color=c, lw=1.3, label=lbl)
            
        ax.set_title(f"{d['name']} (Manifold {m_idx+1})", fontweight='bold', fontsize=10.5)
        ax.set_xlabel("Detuning (MHz)")
        ax.set_ylabel("Diff Absorption (V)")
        ax.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
        ax.grid(True, linestyle=':', alpha=0.55)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

os.makedirs('figures/saturation', exist_ok=True)
windows1 = [(3.7, 5.2), (7.4, 8.1), (15.4, 15.9), (21.8, 22.7)]
windows5 = [(3.3, 4.7), (7.0, 7.7), (14.9, 15.4), (21.2, 22.1)]

slices1 = prepare_manifold_slices('data/Session 9/F1.CSV', 'data/Session 9/P1.CSV', 'data/Session 9/U1.CSV', windows1)
fits1 = run_all_fits(slices1)

slices5 = prepare_manifold_slices('data/Session 9/F5.CSV', 'data/Session 9/P5.CSV', 'data/Session 9/U5.CSV', windows5)
fits5 = run_all_fits(slices5)

make_protocol_plot(1, slices1, fits1, 'figures/saturation/prot1_fits_comparison.png')
make_protocol_plot(5, slices5, fits5, 'figures/saturation/prot5_fits_comparison.png')
make_component_decomposition_plot(1, slices1, fits1, 'figures/saturation/prot1_components_decomposed.png')
make_component_decomposition_plot(5, slices5, fits5, 'figures/saturation/prot5_components_decomposed.png')

# Also make a summary comparison plot of Gamma_0, s, and A/B parameters
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 1. Saturation parameter s comparison
ax = axes[0]
categories = ['Prot 1\n(High Power)', 'Prot 5\n(Low Power)']
s_beam_vals = [fits1['M3_fit'].x[17], fits5['M3_fit'].x[17]]
s_87_vals = [fits1['M2_fit87'].x[9], fits5['M2_fit87'].x[9]]
s_85_vals = [fits1['M2_fit85'].x[9], fits5['M2_fit85'].x[9]]

x_bar = np.arange(len(categories))
width = 0.25
ax.bar(x_bar - width, s_beam_vals, width, label='Global $s_{beam}$', color='#2ca02c', alpha=0.85)
ax.bar(x_bar, s_87_vals, width, label='Isotope $^{87}\\mathrm{Rb}$ $s_{87}$', color='#1f77b4', alpha=0.85)
ax.bar(x_bar + width, s_85_vals, width, label='Isotope $^{85}\\mathrm{Rb}$ $s_{85}$', color='#ff7f0e', alpha=0.85)
ax.set_xticks(x_bar)
ax.set_xticklabels(categories)
ax.set_ylabel("Saturation Parameter $s = I / I_{sat}$")
ax.set_title("Saturation Parameter Convergence", fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.6)

# 2. Gamma_0 baseline vs Natural Linewidth
ax = axes[1]
g0_beam = [fits1['M3_fit'].x[16], fits5['M3_fit'].x[16]]
g0_87 = [fits1['M2_fit87'].x[8], fits5['M2_fit87'].x[8]]
g0_85 = [fits1['M2_fit85'].x[8], fits5['M2_fit85'].x[8]]

ax.bar(x_bar - width, g0_beam, width, label='Global $\\Gamma_0$', color='#2ca02c', alpha=0.85)
ax.bar(x_bar, g0_87, width, label='$\\Gamma_{0, 87}$', color='#1f77b4', alpha=0.85)
ax.bar(x_bar + width, g0_85, width, label='$\\Gamma_{0, 85}$', color='#ff7f0e', alpha=0.85)
ax.axhline(6.066, color='red', linestyle='--', lw=1.8, label='Natural $\\Gamma_{nat} = 6.07\\text{ MHz}$')
ax.set_xticks(x_bar)
ax.set_xticklabels(categories)
ax.set_ylabel("Base Linewidth $\\Gamma_0$ (MHz)")
ax.set_title("Fitted $\\Gamma_0$ vs Natural Linewidth", fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.6)

# 3. Hyperfine Constant A compared to Literature
ax = axes[2]
lit_A = [84.72, 25.00]
fit_A_87 = [fits1['M3_fit'].x[1], fits5['M3_fit'].x[1]]
fit_A_85 = [fits1['M3_fit'].x[5], fits5['M3_fit'].x[5]]
ax.axhline(84.72, color='#1f77b4', linestyle=':', label='Lit $^{87}\\mathrm{Rb}$ A (84.72 MHz)')
ax.plot([0, 1], fit_A_87, 'o-', color='#1f77b4', lw=2, label='Fitted $^{87}\\mathrm{Rb } F=2$')
ax.axhline(25.00, color='#ff7f0e', linestyle=':', label='Lit $^{85}\\mathrm{Rb}$ A (25.00 MHz)')
ax.plot([0, 1], fit_A_85, 's-', color='#ff7f0e', lw=2, label='Fitted $^{85}\\mathrm{Rb } F=3$')
ax.set_xticks([0, 1])
ax.set_xticklabels(categories)
ax.set_ylabel("Magnetic Dipole Constant A (MHz)")
ax.set_title("Hyperfine Accuracy", fontweight='bold')
ax.legend(loc='center right', framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
summary_path = 'figures/saturation/saturation_and_linewidth_summary.png'
plt.savefig(summary_path, dpi=200)
plt.close()
print(f"Saved: {summary_path}")
