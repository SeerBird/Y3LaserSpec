import os
import sys
sys.path.insert(0, '.')
import main
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks

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

WINDOWS_ALL = {
    1: [(3.7, 5.2), (7.4, 8.1), (15.4, 15.9), (21.8, 22.7)],
    2: [(1.9, 3.3), (5.5, 6.2), (13.4, 14.0), (19.9, 20.8)],
    3: [(1.9, 3.2), (5.5, 6.2), (13.4, 14.0), (19.8, 20.7)],
    4: [(2.4, 3.8), (6.1, 6.8), (14.0, 14.5), (20.47, 21.18)],
    5: [(3.3, 4.7), (7.0, 7.7), (14.9, 15.4), (21.2, 22.1)],
    6: [(2.4, 3.7), (6.0, 6.7), (13.9, 14.5), (20.3, 21.2)],
    7: [(2.7, 4.0), (6.3, 7.0), (14.2, 14.7), (20.5, 21.5)],
}

def prepare_slices(prot_idx, fsr=374.74):
    fp = f"data/Session 9/F{prot_idx}.CSV"
    p = f"data/Session 9/P{prot_idx}.CSV"
    u = f"data/Session 9/U{prot_idx}.CSV"
    windows = WINDOWS_ALL[prot_idx]
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
            'C': COUPLINGS[cfg[0]], 'nu_ref_init': nu_ref_init,
            'bg_init': bg_init, 'amp_init': amp_init, 'ref_idx': ref_idx
        })
    return slices

def fit_all_three_models(slices):
    # Model 0: 4 independent Gammas (one uniform per manifold)
    m0_fits = []
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
        fit = least_squares(res, p0, bounds=([x[0]-50, 5, 0, 8.0, -1.0] + [0.0]*6,
                                             [x[-1]+50, 150, 60, 35.0, 1.0] + [5.0]*6))
        def eval_fn(x_in, p=fit.x, cfg=cfg, ref_idx=ref_idx):
            nu_ref, A, B, gamma, bg = p[:5]
            amps = p[5:]
            E_all = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
            offsets = E_all - E_all[ref_idx]
            y_eval = np.full_like(x_in, bg)
            for k in range(6):
                y_eval += amps[k] / (1.0 + 4.0 * ((x_in - (nu_ref + offsets[k])) / gamma)**2)
            return y_eval
        m0_fits.append({'fit': fit, 'eval': eval_fn, 'gamma': fit.x[3], 'A': fit.x[1], 'B': fit.x[2], 'cost': fit.cost})

    # Model 2: Per-Isotope Shared (Gamma_87, s_87 and Gamma_85, s_85)
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
    fit87 = least_squares(res_iso87, p0_87, bounds=([d0['x'][0]-50, 5, 0, -1, d3['x'][0]-50, 5, 0, -1, 5.0, 0.0] + [0.0]*12,
                                                     [d0['x'][-1]+50, 150, 60, 1, d3['x'][-1]+50, 150, 60, 1, 35.0, 20.0] + [5.0]*12))

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
    fit85 = least_squares(res_iso85, p0_85, bounds=([d1['x'][0]-50, 5, 0, -1, d2['x'][0]-50, 5, 0, -1, 5.0, 0.0] + [0.0]*12,
                                                     [d1['x'][-1]+50, 150, 60, 1, d2['x'][-1]+50, 150, 60, 1, 35.0, 20.0] + [5.0]*12))

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

    # Model 3: Globally Shared across both isotopes
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
        return y_eval

    return {
        'M0': m0_fits,
        'M2': {'fit87': fit87, 'fit85': fit85, 'eval': eval_iso},
        'M3': {'fit': fit_glob, 'eval': eval_glob}
    }

def analyze_all_protocols():
    selected_protocols = [1, 3, 5, 7]
    prot_names = {
        1: "Prot 1 (Max Power)",
        3: "Prot 3 (Medium-High Power)",
        5: "Prot 5 (Medium-Low Power)",
        7: "Prot 7 (Lowest Power)"
    }
    
    os.makedirs('figures/protocols_models', exist_ok=True)
    all_results = {}
    
    for p_num in selected_protocols:
        print(f"\n=======================================================")
        print(f"       Fitting {prot_names[p_num]}...")
        print(f"=======================================================")
        slices = prepare_slices(p_num)
        fits = fit_all_three_models(slices)
        all_results[p_num] = {'slices': slices, 'fits': fits}
        
        # Print summary
        m0 = fits['M0']
        m2 = fits['M2']
        m3 = fits['M3']['fit']
        
        print(f"Model 0 (4 Uniform Gammas): Cost = {sum(m['cost'] for m in m0):.4f}")
        for idx, m in enumerate(m0):
            print(f"  {slices[idx]['name']:10s}: Gamma = {m['gamma']:5.2f} MHz | A = {m['A']:6.2f}, B = {m['B']:5.2f}")
            
        print(f"\nModel 2 (Per-Isotope Shared): Cost = {m2['fit87'].cost + m2['fit85'].cost:.4f}")
        print(f"  Rb87: Gamma_0 = {m2['fit87'].x[8]:5.2f} MHz, s = {m2['fit87'].x[9]:.3f} | 87Rb F=2: A={m2['fit87'].x[1]:.2f}, B={m2['fit87'].x[2]:.2f} | 87Rb F=1: A={m2['fit87'].x[5]:.2f}, B={m2['fit87'].x[6]:.2f}")
        print(f"  Rb85: Gamma_0 = {m2['fit85'].x[8]:5.2f} MHz, s = {m2['fit85'].x[9]:.3f} | 85Rb F=3: A={m2['fit85'].x[1]:.2f}, B={m2['fit85'].x[2]:.2f} | 85Rb F=2: A={m2['fit85'].x[5]:.2f}, B={m2['fit85'].x[6]:.2f}")
        
        print(f"\nModel 3 (Globally Shared): Cost = {m3.cost:.4f}")
        print(f"  Global: Gamma_0 = {m3.x[16]:5.2f} MHz, s_beam = {m3.x[17]:.3f}")
        for idx in range(4):
            print(f"  {slices[idx]['name']:10s}: A = {m3.x[idx*4+1]:6.2f}, B = {m3.x[idx*4+2]:5.2f}")
            
        # Plot full 4-manifold comparison for each protocol
        fig, axes = plt.subplots(4, 2, figsize=(15, 11), gridspec_kw={'width_ratios': [2.2, 1.0], 'hspace': 0.38, 'wspace': 0.22})
        fig.suptitle(f"{prot_names[p_num]}: Hyperfine Multiplet Fits Comparison\n"
                     f"M0 (4 $\\Gamma$) vs. M2 Per-Isotope (2 $\\Gamma_0, 2 s$) vs. M3 Global (1 $\\Gamma_0, 1 s$)",
                     fontsize=12, fontweight='bold')
                     
        for m_idx, d in enumerate(slices):
            ax_fit, ax_res = axes[m_idx, 0], axes[m_idx, 1]
            x, y = d['x'], d['y']
            x_dense = np.linspace(x[0], x[-1], 600)
            
            y_m0 = m0[m_idx]['eval'](x)
            y_m0_d = m0[m_idx]['eval'](x_dense)
            res_m0 = y - y_m0
            
            y_m2 = m2['eval'](m_idx, x)
            y_m2_d = m2['eval'](m_idx, x_dense)
            res_m2 = y - y_m2
            
            y_m3 = fits['M3']['eval'](m_idx, x)
            y_m3_d = fits['M3']['eval'](m_idx, x_dense)
            res_m3 = y - y_m3
            
            ax_fit.plot(x, y, 'k.', markersize=2.5, alpha=0.35, label='Data')
            ax_fit.plot(x_dense, y_m0_d, '--', color='#1f77b4', lw=1.3, label=f"M0 Uniform ($\\Gamma={m0[m_idx]['gamma']:.1f}$)")
            ax_fit.plot(x_dense, y_m2_d, '-', color='#2ca02c', lw=1.6, label=f"M2 Per-Isotope")
            ax_fit.plot(x_dense, y_m3_d, ':', color='#d62728', lw=1.8, label=f"M3 Global ($s={m3.x[17]:.2f}$)")
            
            ax_fit.set_title(f"Manifold {m_idx+1}: {d['name']}", fontweight='bold', fontsize=10, loc='left')
            ax_fit.set_ylabel("Diff Abs (V)")
            ax_fit.legend(loc='upper right', framealpha=0.9, fontsize=8)
            ax_fit.grid(True, linestyle=':', alpha=0.55)
            
            ax_res.plot(x, res_m0, color='#1f77b4', lw=0.9, alpha=0.7, label=f"M0 (RMS: {np.std(res_m0)*1e3:.1f} mV)")
            ax_res.plot(x, res_m2, color='#2ca02c', lw=1.0, alpha=0.8, label=f"M2 (RMS: {np.std(res_m2)*1e3:.1f} mV)")
            ax_res.plot(x, res_m3, color='#d62728', lw=0.9, linestyle='--', alpha=0.7, label=f"M3 (RMS: {np.std(res_m3)*1e3:.1f} mV)")
            ax_res.axhline(0, color='gray', linestyle='--', lw=0.8)
            ax_res.set_ylabel("Residual (V)")
            ax_res.set_title(f"Residuals ({d['name']})", fontsize=10, loc='left')
            ax_res.legend(loc='upper right', framealpha=0.9, fontsize=8)
            ax_res.grid(True, linestyle=':', alpha=0.55)
            
            if m_idx == 3:
                ax_fit.set_xlabel("Frequency (MHz)")
                ax_res.set_xlabel("Frequency (MHz)")
                
        plot_path = f"figures/protocols_models/prot{p_num}_three_models_comparison.png"
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved: {plot_path}")

    # Make a grand multi-protocol trend plot of s, Gamma_0, and Cost!
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    prots = selected_protocols
    
    # 1. Saturation parameter s vs Protocol
    ax = axes[0]
    s_beam_trend = [all_results[p]['fits']['M3']['fit'].x[17] for p in prots]
    s_87_trend = [all_results[p]['fits']['M2']['fit87'].x[9] for p in prots]
    s_85_trend = [all_results[p]['fits']['M2']['fit85'].x[9] for p in prots]
    
    ax.plot(prots, s_beam_trend, 'o-', color='#d62728', lw=2, label='Global $s_{beam}$')
    ax.plot(prots, s_87_trend, 's-', color='#1f77b4', lw=1.8, label='$^{87}\\mathrm{Rb}$ $s_{87}$')
    ax.plot(prots, s_85_trend, '^-', color='#2ca02c', lw=1.8, label='$^{85}\\mathrm{Rb}$ $s_{85}$')
    ax.set_xlabel("Protocol (Decreasing Laser Power ->)", fontweight='bold')
    ax.set_ylabel("Saturation Parameter $s = I / I_{sat}$", fontweight='bold')
    ax.set_title("Saturation Parameter vs. Beam Power", fontweight='bold')
    ax.set_xticks(prots)
    ax.set_xticklabels([f"Prot {p}" for p in prots])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Base Linewidth Gamma_0 vs Protocol
    ax = axes[1]
    g0_beam_trend = [all_results[p]['fits']['M3']['fit'].x[16] for p in prots]
    g0_87_trend = [all_results[p]['fits']['M2']['fit87'].x[8] for p in prots]
    g0_85_trend = [all_results[p]['fits']['M2']['fit85'].x[8] for p in prots]
    
    ax.plot(prots, g0_beam_trend, 'o-', color='#d62728', lw=2, label='Global $\\Gamma_0$')
    ax.plot(prots, g0_87_trend, 's-', color='#1f77b4', lw=1.8, label='$^{87}\\mathrm{Rb}$ $\\Gamma_{0, 87}$')
    ax.plot(prots, g0_85_trend, '^-', color='#2ca02c', lw=1.8, label='$^{85}\\mathrm{Rb}$ $\\Gamma_{0, 85}$')
    ax.axhline(6.066, color='black', linestyle='--', lw=1.5, label='Natural $\\Gamma_{nat} = 6.07$ MHz')
    ax.set_xlabel("Protocol (Decreasing Laser Power ->)", fontweight='bold')
    ax.set_ylabel("Baseline Linewidth $\\Gamma_0$ (MHz)", fontweight='bold')
    ax.set_title("Base Linewidth $\\Gamma_0$ vs. Protocol", fontweight='bold')
    ax.set_xticks(prots)
    ax.set_xticklabels([f"Prot {p}" for p in prots])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 3. Fit Residual Cost vs Protocol
    ax = axes[2]
    cost_m0_trend = [sum(m['cost'] for m in all_results[p]['fits']['M0']) for p in prots]
    cost_m2_trend = [all_results[p]['fits']['M2']['fit87'].cost + all_results[p]['fits']['M2']['fit85'].cost for p in prots]
    cost_m3_trend = [all_results[p]['fits']['M3']['fit'].cost for p in prots]
    
    ax.plot(prots, cost_m0_trend, 's--', color='#1f77b4', lw=1.6, label='M0 (4 Independent $\\Gamma$)')
    ax.plot(prots, cost_m2_trend, 'o-', color='#2ca02c', lw=2, label='M2 (Per-Isotope Shared)')
    ax.plot(prots, cost_m3_trend, '^:', color='#d62728', lw=2, label='M3 (Globally Shared)')
    ax.set_xlabel("Protocol (Decreasing Laser Power ->)", fontweight='bold')
    ax.set_ylabel("Sum of Squared Residuals (SSR)", fontweight='bold')
    ax.set_yscale('log')
    ax.set_title("Fit Quality (SSR) vs. Protocol", fontweight='bold')
    ax.set_xticks(prots)
    ax.set_xticklabels([f"Prot {p}" for p in prots])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    trend_path = "figures/protocols_models/multi_protocol_trends.png"
    plt.tight_layout()
    plt.savefig(trend_path, dpi=200)
    plt.close()
    print(f"Saved: {trend_path}")

if __name__ == '__main__':
    analyze_all_protocols()
