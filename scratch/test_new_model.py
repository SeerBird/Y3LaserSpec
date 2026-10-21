import sys
sys.path.insert(0, r"c:\Users\ilyag\Desktop\G\Offline\Code\Y3LaserSpec")
import numpy as np
import physics_model as pm
import time
from scipy.signal import find_peaks

def _solve_populations(R, gamma_0, b_decay, gamma_tr, Ng, Ne):
    prefix = R.shape[:-2]
    N_tot = Ng + Ne
    M = np.zeros((*prefix, N_tot, N_tot))

    sum_R_e = np.sum(R, axis=-1)
    sum_R_g = np.sum(R, axis=-2)

    for g in range(Ng):
        M[..., g, g] = sum_R_e[..., g] + gamma_tr
    for e in range(Ne):
        M[..., Ng + e, Ng + e] = sum_R_g[..., e] + gamma_0 + gamma_tr

    M[..., :Ng, Ng:] = -(R + gamma_0 * b_decay.T)
    M[..., Ng:, :Ng] = -np.swapaxes(R, -2, -1)

    rhs = np.zeros((*prefix, N_tot, 1))
    rhs[..., :Ng, 0] = gamma_tr / float(Ng)

    sol = np.linalg.solve(M, rhs)[..., :, 0]
    return sol[..., :Ng], sol[..., Ng:]

# Test 87Rb F=2 feature peaks
m = pm.MANIFOLDS[0]
nu = np.linspace(-550.0, 100.0, 400)

atom = m.atom_data
Ng, Ne = atom.Ng, atom.Ne
gamma_0 = atom.natural_gamma_mhz
b_decay = atom.b_decay

# Run through simulation
k = 1.380649e-23
T = 295.0
v_th = np.sqrt(2.0 * k * T / atom.mass)
v_z = np.linspace(-350.0, 350.0, 151)
f_v = (1.0 / (np.sqrt(np.pi) * v_th)) * np.exp(-(v_z / v_th) ** 2)
kvz = (atom.k_wave * v_z) / (2.0 * np.pi * 1e6)

nu_grid = nu[:, None, None, None]
kvz_grid = kvz[None, :, None, None]
E_ref = np.mean(atom.E_e[atom.excited_F == m.F_ref])
f_grid = (atom.E_e - E_ref)[None, None, None, :]

target_mask = (atom.ground_F == m.F_ground)
p_q_pump = np.array([0.5, 0.0, 0.5])
p_q_probe = np.array([0.5, 0.0, 0.5])

det_pump = nu_grid - kvz_grid - f_grid
L_pump = 1.0 / (1.0 + 4.0 * (det_pump / 15.0) ** 2)
strength_pump = np.sum(p_q_pump[:, None, None] * (atom.dq ** 2), axis=0) * target_mask[:, None]
R_pump = 0.5 * gamma_0 * 1.5 * strength_pump[None, None, :, :] * L_pump

det_probe = nu_grid + kvz_grid - f_grid
L_probe = 1.0 / (1.0 + 4.0 * (det_probe / 15.0) ** 2)
strength_probe = np.sum(p_q_probe[:, None, None] * (atom.dq ** 2), axis=0) * target_mask[:, None]
R_probe = 0.5 * gamma_0 * 0.0 * strength_probe[None, None, :, :] * L_probe

R_total = R_pump + R_probe
n_g_p, n_e_p = _solve_populations(R_total, gamma_0, b_decay, 0.1, Ng, Ne)
n_g_u = np.ones((*nu.shape, *v_z.shape, Ng)) / float(Ng)
n_e_u = np.zeros((*nu.shape, *v_z.shape, Ne))

R_probe_sensor = strength_probe[None, None, :, :] * L_probe
alpha_v_p = np.einsum('uvge,uvge->uv', n_g_p[:, :, :, None] - n_e_p[:, :, None, :], R_probe_sensor)
alpha_v_u = np.einsum('uvge,uvge->uv', n_g_u[:, :, :, None] - n_e_u[:, :, None, :], R_probe_sensor)

alpha_p = np.trapezoid(alpha_v_p * f_v[None, :], v_z, axis=-1)
alpha_u = np.trapezoid(alpha_v_u * f_v[None, :], v_z, axis=-1)
diff = alpha_u - alpha_p

peaks, _ = find_peaks(diff, prominence=2e-4)
detected_freqs = nu[peaks]
print(f"Detected {len(detected_freqs)} peaks: {detected_freqs}")
expected = np.array([-423.6, -345.1, -266.65, -211.80, -133.33, 0.0])
for exp_peak in expected:
    min_dist = np.min(np.abs(detected_freqs - exp_peak))
    print(f"Peak near {exp_peak}: closest is {min_dist:.2f} MHz away")
