"""
Interactive viewer for saturated absorption power-broadened Lorentzian fits.
Compares:
1. Model 0: Single uniform Gamma per manifold (4 independent Gammas)
2. Model 2: Shared saturation parameter per isotope (Gamma_87, s_87 and Gamma_85, s_85)
3. Model 3: Globally shared beam saturation parameter (single Gamma_0 and s_beam for both isotopes)

Run with: python show_saturation_fits.py
"""

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
            'C': COUPLINGS[cfg[0]], 'nu_ref_init': nu_ref_init,
            'bg_init': bg_init, 'amp_init': amp_init, 'ref_idx': ref_idx
        })
    return slices

def run_all_fits(slices):
    # Model 0: Uniform per manifold
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
                                             [x[-1]+50, 150, 60, 32.0, 1.0] + [5.0]*6))
        def eval_fn(x_in, p=fit.x):
            nu_ref, A, B, gamma, bg = p[:5]
            amps = p[5:]
            E_all = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
            offsets = E_all - E_all[ref_idx]
            y_eval = np.full_like(x_in, bg)
            for k in range(6):
                y_eval += amps[k] / (1.0 + 4.0 * ((x_in - (nu_ref + offsets[k])) / gamma)**2)
            return y_eval
        m0_fits.append({'fit': fit, 'eval': eval_fn, 'gamma': fit.x[3], 'A': fit.x[1], 'B': fit.x[2]})

    # Model 2: Shared per isotope (Gamma_87, s_87 for Rb87; Gamma_85, s_85 for Rb85)
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
            return y_out, g0, s, A0, B0
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
            return y_out, g0, s, A3, B3
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
            return y_out, g0, s, A1, B1
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
            return y_out, g0, s, A2, B2

    # Model 3: Global across both isotopes
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
        comps = []
        for k in range(6):
            comp = (amps[k] * d['C'][k]) / (1.0 + 4.0 * ((x_in - (nu_ref + offsets[k])) / gammas[k])**2)
            y_eval += comp
            comps.append(comp)
        return y_eval, bg, comps, nu_ref + offsets, gammas

    return m0_fits, (fit87, fit85, eval_iso), (fit_glob, eval_glob)

def show_interactive(prot_num, windows):
    fp = f"data/Session 9/F{prot_num}.CSV"
    p = f"data/Session 9/P{prot_num}.CSV"
    u = f"data/Session 9/U{prot_num}.CSV"
    slices = prepare_slices(fp, p, u, windows)
    m0_fits, (fit87, fit85, eval_iso), (fit_glob, eval_glob) = run_all_fits(slices)
    
    g0_glob = fit_glob.x[16]
    s_glob = fit_glob.x[17]
    g0_87, s_87 = fit87.x[8], fit87.x[9]
    g0_85, s_85 = fit85.x[8], fit85.x[9]

    fig, axes = plt.subplots(4, 2, figsize=(15, 11), gridspec_kw={'width_ratios': [2.2, 1.0], 'hspace': 0.38, 'wspace': 0.22})
    fig.canvas.manager.set_window_title(f"Protocol {prot_num}: Saturation Fit Comparison (Per-Isotope vs Global)")
    fig.suptitle(f"Protocol {prot_num}: Hyperfine Fits Comparison\n"
                 f"Per-Isotope: $^{{87}}\\mathrm{{Rb}}$ ($\\Gamma_0={g0_87:.1f}, s={s_87:.2f}$), $^{{85}}\\mathrm{{Rb}}$ ($\\Gamma_0={g0_85:.1f}, s={s_85:.2f}$) | "
                 f"Global: $\\Gamma_0={g0_glob:.1f}\\text{{ MHz}}, s={s_glob:.2f}$",
                 fontsize=11.5, fontweight='bold')

    for m_idx, d in enumerate(slices):
        ax_fit, ax_res = axes[m_idx, 0], axes[m_idx, 1]
        x, y = d['x'], d['y']
        x_dense = np.linspace(x[0], x[-1], 600)

        # M0
        y_m0_dense = m0_fits[m_idx]['eval'](x_dense)
        y_m0_pts = m0_fits[m_idx]['eval'](x)
        res_m0 = y_m0_pts - y

        # M2 Shared per isotope
        y_m2_dense, g0_iso, s_iso, A_iso, B_iso = eval_iso(m_idx, x_dense)
        y_m2_pts, _, _, _, _ = eval_iso(m_idx, x)
        res_m2 = y_m2_pts - y

        # M3 Global across both isotopes
        y_m3_dense, bg_dense, comps, peak_locs, gammas = eval_glob(m_idx, x_dense)
        y_m3_pts, _, _, _, _ = eval_glob(m_idx, x)
        res_m3 = y_m3_pts - y

        ax_fit.plot(x, y, 'k.', markersize=2.5, alpha=0.40, label='Data (P - U)')
        ax_fit.plot(x_dense, y_m0_dense, '--', color='#1f77b4', lw=1.3,
                    label=f"M0 Uniform (4 $\\Gamma$) | $\\Gamma={m0_fits[m_idx]['fit'].x[3]:.1f}\\text{{ MHz}}$")
        ax_fit.plot(x_dense, y_m2_dense, '-', color='#2ca02c', lw=1.6,
                    label=f"M2 Per-Isotope (2 $\\Gamma, 2 s$) | $\\Gamma_0={g0_iso:.1f}, s={s_iso:.2f}$ (A={A_iso:.1f})")
        ax_fit.plot(x_dense, y_m3_dense, ':', color='#d62728', lw=1.8,
                    label=f"M3 Global (1 $\\Gamma, 1 s$) | $\\Gamma_0={g0_glob:.1f}, s={s_glob:.2f}$ (A={fit_glob.x[m_idx*4 + 1]:.1f})")

        pnames = PEAK_NAMES[d['name']]
        y_max = np.max(y)
        for k_pk in range(6):
            c_line = 'navy' if k_pk < 3 else 'darkgreen'
            y_txt = y_max * (0.90 if k_pk < 3 else 0.72)
            ax_fit.axvline(peak_locs[k_pk], color=c_line, linestyle=':', alpha=0.45, lw=0.9)
            ax_fit.text(peak_locs[k_pk], y_txt, pnames[k_pk], color=c_line, fontsize=7.0,
                        ha='center', rotation=90, bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

        ax_fit.set_title(f"Manifold {m_idx+1}: {d['name']}", fontweight='bold', fontsize=10, loc='left')
        ax_fit.set_ylabel("Diff Abs (V)")
        ax_fit.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
        ax_fit.grid(True, linestyle=':', alpha=0.6)

        ax_res.plot(x, res_m0, color='#1f77b4', lw=0.9, alpha=0.7, label=f"M0 Uniform (RMS: {np.std(res_m0)*1e3:.1f} mV)")
        ax_res.plot(x, res_m2, color='#2ca02c', lw=1.0, alpha=0.8, label=f"M2 Per-Isotope (RMS: {np.std(res_m2)*1e3:.1f} mV)")
        ax_res.plot(x, res_m3, color='#d62728', lw=0.9, linestyle='--', alpha=0.7, label=f"M3 Global (RMS: {np.std(res_m3)*1e3:.1f} mV)")
        ax_res.axhline(0, color='gray', linestyle='--', lw=0.8)
        ax_res.set_ylabel("Residual (V)")
        ax_res.set_title(f"Residuals ({d['name']})", fontsize=10, loc='left')
        ax_res.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
        ax_res.grid(True, linestyle=':', alpha=0.6)

        if m_idx == 3:
            ax_fit.set_xlabel("Frequency relative to FSR (MHz)")
            ax_res.set_xlabel("Frequency (MHz)")

if __name__ == '__main__':
    windows1 = [(3.7, 5.2), (7.4, 8.1), (15.4, 15.9), (21.8, 22.7)]
    windows5 = [(3.3, 4.7), (7.0, 7.7), (14.9, 15.4), (21.2, 22.1)]
    print("Opening interactive fit inspection window for Protocol 1...")
    show_interactive(1, windows1)
    print("Opening interactive fit inspection window for Protocol 5...")
    show_interactive(5, windows5)
    plt.show()
