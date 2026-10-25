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

def fit_single_uniform_gamma(d):
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
    b_high = [x[-1]+50, 150, 60, 35.0, 1.0] + [5.0]*6
    fit = least_squares(res, p0, bounds=(b_low, b_high), ftol=1e-6, xtol=1e-6)
    
    dof = len(y) - len(p0)
    s_sq = 2.0 * fit.cost / max(1, dof)
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * s_sq
        unc = np.sqrt(np.diagonal(cov))
    except:
        unc = np.zeros_like(fit.x)
        
    return fit, unc

def fit_six_independent_gammas(d):
    cfg = d['cfg']
    ref_idx = d['ref_idx']
    x, y = d['x'], d['y']
    
    # params: [nu_ref, A, B, bg, gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, amp_0, ..., amp_5] (16 params)
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
    
    dof = len(y) - len(p0)
    s_sq = 2.0 * fit.cost / max(1, dof)
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * s_sq
        unc = np.sqrt(np.diagonal(cov))
        # Correlation matrix
        corr = cov / np.outer(unc, unc)
    except:
        unc = np.zeros_like(fit.x)
        corr = np.zeros((len(fit.x), len(fit.x)))
        
    return fit, unc, corr

def run_investigation():
    windows1 = [(3.7, 5.2), (7.4, 8.1), (15.4, 15.9), (21.8, 22.7)]
    windows5 = [(3.3, 4.7), (7.0, 7.7), (14.9, 15.4), (21.2, 22.1)]
    
    for prot_name, fp, p, u, wins in [
        ('Protocol 1 (High Power)', 'data/Session 9/F1.CSV', 'data/Session 9/P1.CSV', 'data/Session 9/U1.CSV', windows1),
        ('Protocol 5 (Lower Power)', 'data/Session 9/F5.CSV', 'data/Session 9/P5.CSV', 'data/Session 9/U5.CSV', windows5)
    ]:
        print(f"\n=======================================================")
        print(f"               {prot_name}")
        print(f"=======================================================")
        slices = prepare_slices(fp, p, u, wins)
        
        for m_idx, d in enumerate(slices):
            name = d['name']
            lit_A, lit_B = LIT_VALUES[name]
            pnames = PEAK_NAMES[name]
            
            fit_1g, unc_1g = fit_single_uniform_gamma(d)
            fit_6g, unc_6g, corr_6g = fit_six_independent_gammas(d)
            
            print(f"\n--- Manifold {m_idx+1}: {name} (Lit A = {lit_A:.2f}, B = {lit_B:.2f} MHz) ---")
            print(f"1-Gamma Model: Cost = {fit_1g.cost:.4f}")
            print(f"  A = {fit_1g.x[1]:6.2f} +/- {unc_1g[1]:.2f} MHz, B = {fit_1g.x[2]:5.2f} +/- {unc_1g[2]:.2f} MHz")
            print(f"  Uniform Gamma = {fit_1g.x[3]:5.2f} +/- {unc_1g[3]:.2f} MHz")
            
            print(f"6-Gamma Model: Cost = {fit_6g.cost:.4f} (Cost drop: {(fit_1g.cost - fit_6g.cost)/fit_1g.cost*100:.1f}%)")
            print(f"  A = {fit_6g.x[1]:6.2f} +/- {unc_6g[1]:.2f} MHz, B = {fit_6g.x[2]:5.2f} +/- {unc_6g[2]:.2f} MHz")
            print(f"  Fitted Linewidths & Amplitudes:")
            for k in range(6):
                gk = fit_6g.x[4+k]
                gk_err = unc_6g[4+k]
                ak = fit_6g.x[10+k]
                ak_err = unc_6g[10+k]
                # correlation between amp_k and gamma_k
                c_ag = corr_6g[10+k, 4+k]
                at_bound = " [BOUND!]" if (gk <= 5.01 or gk >= 44.99) else ""
                print(f"    {pnames[k]:10s}: Gamma = {gk:5.2f} +/- {gk_err:5.2f} MHz{at_bound:9s} | Amp = {ak:6.4f} +/- {ak_err:6.4f} | Corr(A,G) = {c_ag:+.2f}")

if __name__ == '__main__':
    run_investigation()
