import sys
sys.path.insert(0, '.')
import main
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

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
freq1, _, p1, u1 = main.prep_data('data/Session 9/F1.CSV', 'data/Session 9/P1.CSV', 'data/Session 9/U1.CSV', align_shift=True)
diff1 = p1 - u1
windows1 = [(3.7, 5.2), (7.4, 8.1), (15.4, 15.9), (21.8, 22.7)]
fsr = 374.74

# Model 0: Current main.py (Single uniform gamma per manifold)
print("=== Model 0: Current main.py (Single uniform gamma per manifold) ===")
fits0 = main.fit_hyperfine_multiplets(freq1, diff1, windows=windows1, fsr=fsr)
for f in fits0:
    print(f"{f['name']:10s}: A = {f['A']:6.2f} MHz, B = {f['B']:6.2f} MHz, gamma = {f['gamma']:5.2f} MHz")

# Prepare slices
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

# Model 4: Global fit across ALL 4 manifolds with a single shared beam intensity s_beam and shared Gamma_0
print("\n=== Model 3: Globally Shared Laser Beam s_beam & Gamma_0 (All 4 Manifolds) ===")
# params: [nu_ref0, A0, B0, bg0, nu_ref1, A1, B1, bg1, nu_ref2, A2, B2, bg2, nu_ref3, A3, B3, bg3, Gamma_0, s] + 4*6 amps
def res_global(p):
    residuals = []
    g0, s = p[16], p[17]
    for m_idx in range(4):
        d = manifold_data[m_idx]
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

p0_global = []
b_low_global = []
b_high_global = []
for d in manifold_data:
    p0_global.extend([d['nu_ref_init'], d['cfg'][7][0], d['cfg'][7][1], d['bg_init']])
    b_low_global.extend([d['x'][0]-50, 5.0, 0.0, -1.0])
    b_high_global.extend([d['x'][-1]+50, 150.0, 60.0, 1.0])
p0_global.extend([18.0, 0.5])
b_low_global.extend([5.0, 0.0])
b_high_global.extend([35.0, 15.0])
for d in manifold_data:
    p0_global.extend([d['amp_init']]*6)
    b_low_global.extend([0.0]*6)
    b_high_global.extend([5.0]*6)

fit_glob = least_squares(res_global, p0_global, bounds=(b_low_global, b_high_global))
g0_g, s_g = fit_glob.x[16], fit_glob.x[17]
print(f"Global Fit: Gamma_0 = {g0_g:.2f} MHz, s_beam (I/Isat) = {s_g:.2f} | total cost = {fit_glob.cost:.4f}")
for m_idx in range(4):
    d = manifold_data[m_idx]
    A_fit = fit_glob.x[m_idx*4 + 1]
    B_fit = fit_glob.x[m_idx*4 + 2]
    print(f"  {d['name']:10s}: A = {A_fit:6.2f} MHz, B = {B_fit:6.2f} MHz")
