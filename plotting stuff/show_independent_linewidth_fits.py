"""
Interactive viewer for saturated absorption hyperfine multiplet fits 
where EACH linewidth is an independent free parameter (6 independent gammas per manifold).

Displays:
- Full multi-Lorentzian fit vs experimental data
- All 6 individual decomposed Lorentzian components with their respective widths
- Casimir formula peak centers
- Residuals (Data - Fit) with RMS metrics
- Supports Protocol 1 and Protocol 5 (or any protocol 1-7 via argument)

Run with:
  python show_independent_linewidth_fits.py
or
  python show_independent_linewidth_fits.py 1
  python show_independent_linewidth_fits.py 5
"""

import sys
sys.path.insert(0, '..')
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

LIT_VALUES = {
    '87Rb F=2': (84.72, 12.50),
    '85Rb F=3': (25.00, 25.80),
    '85Rb F=2': (25.00, 25.80),
    '87Rb F=1': (84.72, 12.50),
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
            'nu_ref_init': nu_ref_init, 'bg_init': bg_init,
            'amp_init': amp_init, 'ref_idx': ref_idx
        })
    return slices

def fit_6gamma(d):
    cfg, ref_idx, x, y = d['cfg'], d['ref_idx'], d['x'], d['y']
    
    # params: [nu_ref, A, B, bg, gamma_0..5, amp_0..5] (16 params)
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
    b_low = [x[0]-50, 5.0, 0.0, -1.0] + [5.0]*6 + [0.0]*6
    b_high = [x[-1]+50, 150.0, 60.0, 1.0] + [45.0]*6 + [5.0]*6
    fit = least_squares(res, p0, bounds=(b_low, b_high), ftol=1e-6, xtol=1e-6)
    
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
        return y_eval, bg, comps, nu_ref + offsets, gammas, amps
        
    return fit, eval_fn

def show_protocol_independent_fits(prot_idx):
    slices = prepare_slices(prot_idx)
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 11), gridspec_kw={'width_ratios': [2.2, 1.0], 'hspace': 0.40, 'wspace': 0.22})
    fig.canvas.manager.set_window_title(f"Protocol {prot_idx}: Independent Linewidth Fits (6 Free Gammas)")
    fig.suptitle(f"Protocol {prot_idx}: Multi-Lorentzian Fits with Independent Linewidths\n"
                 f"Each transition has its own free $\\Gamma_k$ (Casimir positions constrained by A & B)",
                 fontsize=12, fontweight='bold')
                 
    colors_true = ['#1f77b4', '#2ca02c', '#9467bd']
    colors_co = ['#ff7f0e', '#8c564b', '#e377c2']
    
    print(f"\n=======================================================")
    print(f"       Protocol {prot_idx}: Independent Linewidth Fits")
    print(f"=======================================================")
    
    for m_idx, d in enumerate(slices):
        fit, eval_fn = fit_6gamma(d)
        x, y = d['x'], d['y']
        x_dense = np.linspace(x[0], x[-1], 700)
        
        y_tot, bg, comps, peak_locs, gammas, amps = eval_fn(x_dense)
        y_pts, _, _, _, _, _ = eval_fn(x)
        res = y - y_pts
        
        A_fit, B_fit = fit.x[1], fit.x[2]
        lit_A, lit_B = LIT_VALUES[d['name']]
        pnames = PEAK_NAMES[d['name']]
        
        print(f"\nManifold {m_idx+1}: {d['name']} | Cost = {fit.cost:.4f}")
        print(f"  A = {A_fit:6.2f} MHz (Lit: {lit_A:.2f}), B = {B_fit:5.2f} MHz (Lit: {lit_B:.2f})")
        for k in range(6):
            bound_flag = " [BOUND]" if (gammas[k] <= 5.05 or gammas[k] >= 44.95) else ""
            print(f"    {pnames[k]:10s}: Gamma = {gammas[k]:5.2f} MHz{bound_flag:8s} | Amp = {amps[k]:6.4f} V | Peak = {peak_locs[k]:6.1f} MHz")
            
        ax_fit, ax_res = axes[m_idx, 0], axes[m_idx, 1]
        
        # Left Subplot: Data, Total Fit, and Decomposed Components
        ax_fit.plot(x, y, 'k.', markersize=2.5, alpha=0.35, label='Experimental Data (P - U)')
        ax_fit.plot(x_dense, y_tot, 'r-', lw=2.0, label=f"Total Fit (A={A_fit:.1f}, B={B_fit:.1f})")
        ax_fit.plot(x_dense, np.full_like(x_dense, bg), 'k--', lw=0.9, alpha=0.6, label=f"Baseline ({bg:.3f} V)")
        
        # Plot individual Lorentzian components
        for k in range(6):
            c = colors_true[k] if k < 3 else colors_co[k-3]
            ls = '-' if k < 3 else '--'
            lbl = f"{pnames[k]}: $\\Gamma={gammas[k]:.1f}$ MHz, $A={amps[k]:.3f}$ V"
            ax_fit.plot(x_dense, bg + comps[k], linestyle=ls, color=c, lw=1.2, alpha=0.85, label=lbl)
            
        # Mark peak locations
        y_max = np.max(y)
        for k in range(6):
            c_line = 'navy' if k < 3 else 'darkgreen'
            ax_fit.axvline(peak_locs[k], color=c_line, linestyle=':', alpha=0.45, lw=0.8)
            ax_fit.text(peak_locs[k], y_max * (0.92 if k < 3 else 0.74), f"{pnames[k]}",
                        color=c_line, fontsize=7.0, ha='center', rotation=90,
                        bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))
                        
        ax_fit.set_title(f"Manifold {m_idx+1}: {d['name']} | A={A_fit:.2f}, B={B_fit:.2f} MHz | $\\Gamma_k \\in [{np.min(gammas):.1f}, {np.max(gammas):.1f}]$ MHz",
                         fontweight='bold', fontsize=10, loc='left')
        ax_fit.set_ylabel("Diff Abs (V)", fontsize=9.5)
        ax_fit.grid(True, linestyle=':', alpha=0.55)
        
        # Right Subplot: Residuals
        ax_res.plot(x, res, color='crimson', lw=1.0, alpha=0.8)
        ax_res.axhline(0, color='black', linestyle='--', lw=0.8)
        ax_res.set_ylabel("Residual (V)", fontsize=9.5)
        ax_res.set_title(f"Residuals ({d['name']}) | RMS: {np.std(res)*1e3:.1f} mV", fontsize=9.5, loc='left')
        ax_res.grid(True, linestyle=':', alpha=0.55)
        
        if m_idx == 3:
            ax_fit.set_xlabel("Frequency relative to FSR (MHz)", fontsize=10)
            ax_res.set_xlabel("Frequency (MHz)", fontsize=10)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 0:
        for arg in args:
            try:
                p_num = int(arg)
                show_protocol_independent_fits(p_num)
            except ValueError:
                print(f"Invalid protocol number: {arg}")
    else:
        # Default: open Protocol 1 and Protocol 5
        print("Opening interactive independent linewidth fits for Protocol 1...")
        show_protocol_independent_fits(1)
        print("Opening interactive independent linewidth fits for Protocol 5...")
        show_protocol_independent_fits(5)
        
    print("\nDisplaying interactive matplotlib windows. Use toolbar to zoom/pan.")
    plt.show()
