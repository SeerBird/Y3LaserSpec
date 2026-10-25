import sys
sys.path.insert(0, '.')
import main
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

# Precomputed dipole transition coupling factors C_k for each manifold
# [True F' transitions..., Crossover transitions...] normalized to max = 1.0
COUPLINGS = {
    '87Rb F=2': np.array([0.05714286, 0.28571429, 0.8,        0.07142857, 0.2,        1.0]),
    '85Rb F=3': np.array([0.06349206, 0.22222222, 0.51428571, 0.12345679, 0.28571429, 1.0]),
    '85Rb F=2': np.array([0.49586777, 0.64285714, 0.51428571, 0.96428571, 0.77142857, 1.0]),
    '87Rb F=1': np.array([0.32,       0.8,        0.8,        0.4,        0.4,        1.0]),
}

configs = [
    ('87Rb F=2', 2, 0.5, 1.5, 1.5, [1, 2, 3], 2, (84.7, 12.5)),
    ('85Rb F=3', 3, 0.5, 1.5, 2.5, [2, 3, 4], 2, (25.0, 25.8)),
    ('85Rb F=2', 2, 0.5, 1.5, 2.5, [1, 2, 3], 0, (25.0, 25.8)),
    ('87Rb F=1', 1, 0.5, 1.5, 1.5, [0, 1, 2], 0, (84.7, 12.5)),
]

couplings = [COUPLINGS[cfg[0]] for cfg in configs]

# Load Prot1
freq1, _, p1, u1 = main.prep_data('data/Session 9/F1.CSV', 'data/Session 9/P1.CSV', 'data/Session 9/U1.CSV', align_shift=True)
diff1 = p1 - u1
windows1 = [(3.7, 5.2), (7.4, 8.1), (15.4, 15.9), (21.8, 22.7)]
fsr = 374.74

# Prepare data slices for the 4 manifolds
manifold_data = []
for i, (cfg, (w0, w1)) in enumerate(zip(configs, windows1)):
    m = (freq1 >= w0) & (freq1 <= w1)
    x = freq1[m] * fsr
    y = diff1[m]
    ref_idx = cfg[6]
    A_init, B_init = cfg[7]
    E_init = main.calc_peak_offsets(A_init, B_init, cfg[4], cfg[3], cfg[5])
    offsets_init = E_init - E_init[ref_idx]
    dpeaks, _ = find_peaks(y, prominence=0.01)
    if cfg[0] == '85Rb F=2':
        nu_ref_init = float(x[dpeaks[-1]] - offsets_init[5])
    elif cfg[0] == '87Rb F=1':
        nu_ref_init = float(x[dpeaks[-1]] - offsets_init[2])
    else:
        k_max = np.argmax(offsets_init)
        nu_ref_init = float(x[dpeaks[-1]] - offsets_init[k_max])
    bg_init = float(np.percentile(y, 10))
    amp_init = float(np.max(y) - bg_init) / 3.0
    manifold_data.append({
        'name': cfg[0], 'cfg': cfg, 'x': x, 'y': y, 'C': couplings[i],
        'nu_ref_init': nu_ref_init, 'bg_init': bg_init, 'amp_init': amp_init,
        'ref_idx': ref_idx
    })

# Case 1: Independent s per manifold
# For each manifold: params = [nu_ref, A, B, bg, Gamma_0, s] + 6 amps
print("=== Case 1: Independent s per manifold (Gamma_k = Gamma_0 * sqrt(1 + s * C_k)) ===")
for i, d in enumerate(manifold_data):
    cfg = d['cfg']
    C = d['C']
    ref_idx = d['ref_idx']
    x, y = d['x'], d['y']
    
    def model_indep(p):
        nu_ref, A, B, bg, g0, s = p[:6]
        amps = p[6:]
        E_all = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
        offsets = E_all - E_all[ref_idx]
        gammas = g0 * np.sqrt(1.0 + s * C)
        y_eval = np.full_like(x, bg)
        for k in range(6):
            y_eval += (amps[k] * C[k]) / (1.0 + 4.0 * ((x - (nu_ref + offsets[k])) / gammas[k])**2)
        return y_eval - y
    
    A_guess, B_guess = cfg[7]
    p0 = [d['nu_ref_init'], A_guess, B_guess, d['bg_init'], 15.0, 1.0] + [d['amp_init']]*6
    b_low = [x[0]-50, 5, 0, -1, 5.0, 0.0] + [0.0]*6
    b_high = [x[-1]+50, 150, 60, 1, 35.0, 15.0] + [5.0]*6
    fit = least_squares(model_indep, p0, bounds=(b_low, b_high))
    
    g0_fit, s_fit = fit.x[4], fit.x[5]
    gammas_fit = g0_fit * np.sqrt(1.0 + s_fit * C)
    print(f"{d['name']:10s}: A = {fit.x[1]:6.2f} MHz, B = {fit.x[2]:6.2f} MHz | Gamma_0 = {g0_fit:5.2f} MHz, s (I/Isat) = {s_fit:5.2f} | Gammas = {np.round(gammas_fit, 1)} MHz | cost = {fit.cost:.4f}")

# Case 2: Shared s per isotope!
# Isotope 87: manifolds 0 and 3 share s_87 and Gamma_0_87
# Isotope 85: manifolds 1 and 2 share s_85 and Gamma_0_85
print("\n=== Case 2: Shared s per isotope (s_87 shared by M1 and M4, s_85 shared by M2 and M3) ===")

# Simultaneous fit for Rb87 (M0 and M3):
d0, d3 = manifold_data[0], manifold_data[3]
def res_rb87(p):
    nu_ref0, A0, B0, bg0, nu_ref3, A3, B3, bg3, g0, s = p[:10]
    amps0 = p[10:16]
    amps3 = p[16:22]
    
    # M0
    E0 = main.calc_peak_offsets(A0, B0, d0['cfg'][4], d0['cfg'][3], d0['cfg'][5])
    off0 = E0 - E0[d0['ref_idx']]
    g0_vec = g0 * np.sqrt(1.0 + s * d0['C'])
    y0_eval = np.full_like(d0['x'], bg0)
    for k in range(6):
        y0_eval += (amps0[k] * d0['C'][k]) / (1.0 + 4.0 * ((d0['x'] - (nu_ref0 + off0[k])) / g0_vec[k])**2)
    r0 = y0_eval - d0['y']
    
    # M3
    E3 = main.calc_peak_offsets(A3, B3, d3['cfg'][4], d3['cfg'][3], d3['cfg'][5])
    off3 = E3 - E3[d3['ref_idx']]
    g3_vec = g0 * np.sqrt(1.0 + s * d3['C'])
    y3_eval = np.full_like(d3['x'], bg3)
    for k in range(6):
        y3_eval += (amps3[k] * d3['C'][k]) / (1.0 + 4.0 * ((d3['x'] - (nu_ref3 + off3[k])) / g3_vec[k])**2)
    r3 = y3_eval - d3['y']
    
    return np.concatenate([r0, r3])

p0_87 = [d0['nu_ref_init'], 84.7, 12.5, d0['bg_init'], d3['nu_ref_init'], 84.7, 12.5, d3['bg_init'], 15.0, 1.0] + [d0['amp_init']]*6 + [d3['amp_init']]*6
fit_87 = least_squares(res_rb87, p0_87, bounds=([d0['x'][0]-50, 5, 0, -1, d3['x'][0]-50, 5, 0, -1, 5.0, 0.0] + [0.0]*12,
                                                 [d0['x'][-1]+50, 150, 60, 1, d3['x'][-1]+50, 150, 60, 1, 35.0, 15.0] + [5.0]*12))

g0_87, s_87 = fit_87.x[8], fit_87.x[9]
print(f"Rb87 Shared Fit: Gamma_0 = {g0_87:.2f} MHz, s (I/Isat) = {s_87:.2f} | cost = {fit_87.cost:.4f}")
print(f"  87Rb F=2: A = {fit_87.x[1]:.2f} MHz, B = {fit_87.x[2]:.2f} MHz")
print(f"  87Rb F=1: A = {fit_87.x[5]:.2f} MHz, B = {fit_87.x[6]:.2f} MHz")

# Simultaneous fit for Rb85 (M1 and M2):
d1, d2 = manifold_data[1], manifold_data[2]
def res_rb85(p):
    nu_ref1, A1, B1, bg1, nu_ref2, A2, B2, bg2, g0, s = p[:10]
    amps1 = p[10:16]
    amps2 = p[16:22]
    
    # M1
    E1 = main.calc_peak_offsets(A1, B1, d1['cfg'][4], d1['cfg'][3], d1['cfg'][5])
    off1 = E1 - E1[d1['ref_idx']]
    g1_vec = g0 * np.sqrt(1.0 + s * d1['C'])
    y1_eval = np.full_like(d1['x'], bg1)
    for k in range(6):
        y1_eval += (amps1[k] * d1['C'][k]) / (1.0 + 4.0 * ((d1['x'] - (nu_ref1 + off1[k])) / g1_vec[k])**2)
    r1 = y1_eval - d1['y']
    
    # M2
    E2 = main.calc_peak_offsets(A2, B2, d2['cfg'][4], d2['cfg'][3], d2['cfg'][5])
    off2 = E2 - E2[d2['ref_idx']]
    g2_vec = g0 * np.sqrt(1.0 + s * d2['C'])
    y2_eval = np.full_like(d2['x'], bg2)
    for k in range(6):
        y2_eval += (amps2[k] * d2['C'][k]) / (1.0 + 4.0 * ((d2['x'] - (nu_ref2 + off2[k])) / g2_vec[k])**2)
    r2 = y2_eval - d2['y']
    
    return np.concatenate([r1, r2])

p0_85 = [d1['nu_ref_init'], 25.0, 25.8, d1['bg_init'], d2['nu_ref_init'], 25.0, 25.8, d2['bg_init'], 15.0, 1.0] + [d1['amp_init']]*6 + [d2['amp_init']]*6
fit_85 = least_squares(res_rb85, p0_85, bounds=([d1['x'][0]-50, 5, 0, -1, d2['x'][0]-50, 5, 0, -1, 5.0, 0.0] + [0.0]*12,
                                                 [d1['x'][-1]+50, 150, 60, 1, d2['x'][-1]+50, 150, 60, 1, 35.0, 15.0] + [5.0]*12))

g0_85, s_85 = fit_85.x[8], fit_85.x[9]
print(f"Rb85 Shared Fit: Gamma_0 = {g0_85:.2f} MHz, s (I/Isat) = {s_85:.2f} | cost = {fit_85.cost:.4f}")
print(f"  85Rb F=3: A = {fit_85.x[1]:.2f} MHz, B = {fit_85.x[2]:.2f} MHz")
print(f"  85Rb F=2: A = {fit_85.x[5]:.2f} MHz, B = {fit_85.x[6]:.2f} MHz")

# Case 3: Also test with fixed Gamma_0 = 6.066 MHz (natural linewidth) vs free Gamma_0
print("\n=== Case 3: Natural Linewidth Gamma_0 = 6.07 MHz fixed, fit s (I/Isat) ===")
for i, d in enumerate(manifold_data):
    cfg = d['cfg']
    C = d['C']
    ref_idx = d['ref_idx']
    x, y = d['x'], d['y']
    
    def model_gamma0_fixed(p):
        nu_ref, A, B, bg, s = p[:5]
        amps = p[5:]
        E_all = main.calc_peak_offsets(A, B, cfg[4], cfg[3], cfg[5])
        offsets = E_all - E_all[ref_idx]
        gammas = 6.066 * np.sqrt(1.0 + s * C)
        y_eval = np.full_like(x, bg)
        for k in range(6):
            y_eval += (amps[k] * C[k]) / (1.0 + 4.0 * ((x - (nu_ref + offsets[k])) / gammas[k])**2)
        return y_eval - y
    
    A_guess, B_guess = cfg[7]
    p0 = [d['nu_ref_init'], A_guess, B_guess, d['bg_init'], 5.0] + [d['amp_init']]*6
    b_low = [x[0]-50, 5, 0, -1, 0.0] + [0.0]*6
    b_high = [x[-1]+50, 150, 60, 1, 50.0] + [5.0]*6
    fit = least_squares(model_gamma0_fixed, p0, bounds=(b_low, b_high))
    
    s_fit = fit.x[4]
    gammas_fit = 6.066 * np.sqrt(1.0 + s_fit * C)
    print(f"{d['name']:10s}: A = {fit.x[1]:6.2f} MHz, B = {fit.x[2]:6.2f} MHz | s (I/Isat) = {s_fit:5.2f} | Gammas = {np.round(gammas_fit, 1)} MHz | cost = {fit.cost:.4f}")
