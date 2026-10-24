from dataclasses import dataclass
import numpy as np
from scipy.constants import c, k
from scipy.optimize import least_squares
import pylcp
import pylcp.hamiltonians as h
from sympy.physics.wigner import clebsch_gordan as sympy_cg


@dataclass
class AtomicStructure:
    """
    Encapsulates all atomic states, hyperfine levels, dipole matrix elements,
    and Zeeman shifts for a specific rubidium isotope.
    """
    isotope: str
    atom: object
    mass: float
    k_wave: float
    natural_gamma_mhz: float
    Ng: int
    Ne: int
    ground_F: np.ndarray
    ground_m: np.ndarray
    excited_F: np.ndarray
    excited_m: np.ndarray
    E_g: np.ndarray
    E_e: np.ndarray
    mu_z_g: np.ndarray
    mu_z_e: np.ndarray
    mu_x_mat_g: np.ndarray
    mu_y_mat_g: np.ndarray
    mu_z_mat_g: np.ndarray
    dq: np.ndarray
    b_decay: np.ndarray
    partner_plus: np.ndarray
    partner_minus: np.ndarray

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class AtomRegistry:
    """Registry providing clean object attribute access to rubidium isotopes."""
    rb87: AtomicStructure
    rb85: AtomicStructure

    def __getitem__(self, key: str) -> AtomicStructure:
        key_str = str(key)
        if key_str in ('87', '87Rb', 'Rb87'):
            return self.rb87
        elif key_str in ('85', '85Rb', 'Rb85'):
            return self.rb85
        raise KeyError(f"Unknown isotope: {key}")


def _build_single_atom_structure(name: str, iso: str) -> AtomicStructure:
    """Builds an AtomicStructure instance for a given rubidium isotope using pylcp."""
    atom = pylcp.atom(name)
    g_state = atom.state[0]  # 5^2 S_{1/2} ground state
    e_state = atom.state[2]  # 5^2 P_{3/2} excited state (D2 line)
    d2_trans = atom.transition[1]  # D2 optical transition

    # Hyperfine Hamiltonians and Zeeman matrices
    H0_g, mu_g, basis_g = h.hyperfine_coupled(
        g_state.J, atom.I, g_state.gJ, atom.gI, g_state.Ahfs, g_state.Bhfs, return_basis=True
    )
    H0_e, mu_e, basis_e = h.hyperfine_coupled(
        e_state.J, atom.I, e_state.gJ, atom.gI, e_state.Ahfs, e_state.Bhfs, return_basis=True
    )

    # Normalized transition dipole matrix elements dq[q, g, e]:
    # q=0: delta_m=+1 (sigma+), q=1: delta_m=0 (pi), q=2: delta_m=-1 (sigma-)
    dq, bg, be = h.dqij_two_hyperfine_manifolds(
        g_state.J, e_state.J, atom.I, normalize=True, return_basis=True
    )

    # Spontaneous decay branching ratios b_decay[ie, ig]
    sum_dq2 = np.sum(dq**2, axis=0)  # shape (Ng, Ne)
    total_decay_per_e = np.sum(sum_dq2, axis=0)  # shape (Ne,)
    b_decay = (sum_dq2 / total_decay_per_e[None, :]).T  # shape (Ne, Ng)

    # Hyperfine energies in MHz
    E_g = np.diag(H0_g) * 1e-6
    E_e = np.diag(H0_e) * 1e-6

    # Longitudinal Zeeman shift rates in MHz/Gauss
    mu_z_g = -mu_g[1].diagonal() * 1e-6
    mu_z_e = -mu_e[1].diagonal() * 1e-6

    # Full Cartesian magnetic dipole interaction matrices in MHz/Gauss
    # H_Z = mu_x*Bx + mu_y*By + mu_z*Bz where mu = -mu_pylcp * 1e-6
    mu_x_mat_g = -(mu_g[0] - mu_g[2]) / np.sqrt(2.0) * 1e-6
    mu_y_mat_g = -1j * (mu_g[0] + mu_g[2]) / np.sqrt(2.0) * 1e-6
    mu_z_mat_g = -mu_g[1] * 1e-6

    # Lambda-system partners within each ground hyperfine level:
    # partner_plus[g] is ground state with Delta m = +2 (for sigma+ / sigma- interference)
    # partner_minus[g] is ground state with Delta m = -2
    partner_plus = np.full(len(bg), -1, dtype=int)
    partner_minus = np.full(len(bg), -1, dtype=int)
    for g in range(len(bg)):
        F_val = bg[g, 0]
        m_val = bg[g, 1]
        p_p = np.where((bg[:, 0] == F_val) & (bg[:, 1] == m_val + 2))[0]
        if len(p_p) > 0:
            partner_plus[g] = p_p[0]
        p_m = np.where((bg[:, 0] == F_val) & (bg[:, 1] == m_val - 2))[0]
        if len(p_m) > 0:
            partner_minus[g] = p_m[0]

    return AtomicStructure(
        isotope=iso,
        atom=atom,
        mass=atom.mass,
        k_wave=2.0 * np.pi / d2_trans.lam,
        natural_gamma_mhz=e_state.gammaHz * 1e-6,
        Ng=len(bg),
        Ne=len(be),
        ground_F=bg[:, 0],
        ground_m=bg[:, 1],
        excited_F=be[:, 0],
        excited_m=be[:, 1],
        E_g=E_g,
        E_e=E_e,
        mu_z_g=mu_z_g,
        mu_z_e=mu_z_e,
        mu_x_mat_g=mu_x_mat_g,
        mu_y_mat_g=mu_y_mat_g,
        mu_z_mat_g=mu_z_mat_g,
        dq=dq,
        b_decay=b_decay,
        partner_plus=partner_plus,
        partner_minus=partner_minus,
    )


def _build_atom_structures() -> AtomRegistry:
    return AtomRegistry(
        rb87=_build_single_atom_structure('87Rb', '87'),
        rb85=_build_single_atom_structure('85Rb', '85'),
    )


ATOM_DATA = _build_atom_structures()


@dataclass
class HyperfineManifold:
    """Represents an invariant hyperfine ground-state absorption manifold."""
    name: str
    isotope: str
    F_ground: int
    F_ref: int

    @property
    def atom_data(self) -> AtomicStructure:
        return ATOM_DATA[self.isotope]

    def __getitem__(self, item):
        return getattr(self, item)


# Hyperfine manifolds under study (pure atomic quantum invariants)
MANIFOLDS = [
    HyperfineManifold(name='87Rb F=2', isotope='87', F_ground=2, F_ref=3),
    HyperfineManifold(name='85Rb F=3', isotope='85', F_ground=3, F_ref=4),
    HyperfineManifold(name='85Rb F=2', isotope='85', F_ground=2, F_ref=1),
    HyperfineManifold(name='87Rb F=1', isotope='87', F_ground=1, F_ref=0),
]


def _solve_steady_state_populations(R, gamma_0, b_decay, gamma_tr, Ng, Ne):
    """
    Solves the coupled steady-state rate equations for all ground and excited sublevels
    without adiabatic elimination (Approach C).
    
    R : ndarray of shape (..., Ng, Ne)
        Stimulated optical transition rates.
    b_decay : ndarray of shape (Ne, Ng)
        Spontaneous decay branching ratios from excited state e to ground state g.
    gamma_0 : float
        Natural decay rate (MHz).
    gamma_tr : float
        Transit relaxation rate (MHz).
    """
    prefix = R.shape[:-2]
    N_tot = Ng + Ne
    M = np.zeros((*prefix, N_tot, N_tot))

    sum_R_e = np.sum(R, axis=-1)  # Total excitation rate from ground state g
    sum_R_g = np.sum(R, axis=-2)  # Total excitation rate into excited state e

    # Ground state diagonal (loss via optical excitation + transit)
    for g in range(Ng):
        M[..., g, g] = sum_R_e[..., g] + gamma_tr

    # Excited state diagonal (loss via stimulated emission + spontaneous decay + transit)
    for e in range(Ne):
        M[..., Ng + e, Ng + e] = sum_R_g[..., e] + gamma_0 + gamma_tr

    # Off-diagonal blocks:
    # Inflow to ground states from excited states (stimulated + spontaneous emission)
    M[..., :Ng, Ng:] = -(R + gamma_0 * b_decay.T)
    # Inflow to excited states from ground states (absorption)
    M[..., Ng:, :Ng] = -np.swapaxes(R, -2, -1)

    # RHS: Fresh thermal atoms entering the beam in ground states
    rhs = np.zeros((*prefix, N_tot, 1))
    rhs[..., :Ng, 0] = gamma_tr / float(Ng)

    sol = np.linalg.solve(M, rhs)[..., :, 0]
    return sol[..., :Ng], sol[..., Ng:]


def decompose_laser_polarization(
    Bx: float, By: float, Bz: float, epsilon: float
) -> tuple[float, float, float]:
    """
    Decomposes an elliptical laser polarization (propagating along z, major axis x)
    into (I_plus, I_pi, I_minus) relative to an arbitrary 3D magnetic quantization axis B.
    """
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    if B_mag < 1e-9:
        I_plus = 0.5 * (1.0 + np.sin(2.0 * epsilon))
        I_minus = 0.5 * (1.0 - np.sin(2.0 * epsilon))
        I_pi = 0.0
        return float(I_plus), float(I_pi), float(I_minus)

    cos_eps2 = np.cos(epsilon)**2
    sin_eps2 = np.sin(epsilon)**2
    I_pi = (Bx**2 * cos_eps2 + By**2 * sin_eps2) / (B_mag**2)
    delta_I = np.sin(2.0 * epsilon) * (Bz / B_mag)

    I_plus = 0.5 * (1.0 - I_pi + delta_I)
    I_minus = 0.5 * (1.0 - I_pi - delta_I)
    return max(0.0, float(I_plus)), max(0.0, float(I_pi)), max(0.0, float(I_minus))


def simulate_manifold_coherent_operator(
    nu_local_mhz,
    manifold: HyperfineManifold,
    Bx_gauss,
    By_gauss,
    Bz_gauss,
    epsilon_pump,
    epsilon_probe,
    T,
    s_pump,
    kappa_probe,
    gamma_eff,
    gamma_tr,
    N_v,
):
    """
    Approach 3: Universal Coherent Transition Operator.
    Evaluates coherent dipole vector sum across all transitions universally,
    incorporating two-photon Raman dephasing in arbitrary 3D magnetic field B.
    """
    atom_info = manifold.atom_data
    mass = atom_info.mass
    k_wave = atom_info.k_wave
    gamma_0 = atom_info.natural_gamma_mhz
    F_g_target = manifold.F_ground
    F_ref = manifold.F_ref

    Ng = atom_info.Ng
    Ne = atom_info.Ne
    dq = atom_info.dq  # (3, Ng, Ne)
    b_decay = atom_info.b_decay
    ground_F = atom_info.ground_F
    excited_F = atom_info.excited_F
    E_e = atom_info.E_e
    mu_z_g = atom_info.mu_z_g
    mu_z_e = atom_info.mu_z_e

    E_ref = E_e[excited_F == F_ref][0]
    f_res_local = E_e - E_ref

    B_mag = np.sqrt(Bx_gauss**2 + By_gauss**2 + Bz_gauss**2)

    # Polarization decomposition in the magnetic quantization frame
    I_plus_pump, I_pi_pump, I_minus_pump = decompose_laser_polarization(
        Bx_gauss, By_gauss, Bz_gauss, epsilon_pump
    )
    I_plus_probe, I_pi_probe, I_minus_probe = decompose_laser_polarization(
        Bx_gauss, By_gauss, Bz_gauss, epsilon_probe
    )

    # Ground-state Lande factor g_F for target manifold:
    # 87Rb: F=1 => g_F = -1/2, F=2 => g_F = +1/2
    # 85Rb: F=2 => g_F = -1/3, F=3 => g_F = +1/3
    g_F_val = (0.5 if F_g_target == 2 else 0.5) if manifold.isotope == '87' else (1.0 / 3.0)
    delta_R = 2.0 * g_F_val * 1.3996 * B_mag  # MHz
    D_Raman = 1.0 / (1.0 + (delta_R / gamma_tr) ** 2) if gamma_tr > 0 else 1.0

    target_mask = (ground_F == F_g_target)[:, None]

    incoherent_pump = (
        I_plus_pump * dq[0]**2 +
        I_pi_pump * dq[1]**2 +
        I_minus_pump * dq[2]**2
    )
    strength_pump = incoherent_pump * target_mask

    incoherent_probe = (
        I_plus_probe * dq[0]**2 +
        I_pi_probe * dq[1]**2 +
        I_minus_probe * dq[2]**2
    )
    strength_probe = incoherent_probe * target_mask

    v_th = np.sqrt(2.0 * k * T / mass)
    v_max = 350.0  # m/s
    v_z = np.linspace(-v_max, v_max, N_v)
    f_v = (1.0 / (np.sqrt(np.pi) * v_th)) * np.exp(-(v_z / v_th) ** 2)
    kvz = (k_wave * v_z) / (2.0 * np.pi * 1e6)

    nu_grid = nu_local_mhz[:, None, None, None]
    kvz_grid = kvz[None, :, None, None]
    f_grid = f_res_local[None, None, None, :]

    if B_mag > 1e-6:
        delta_Z = (mu_z_e[None, :] - mu_z_g[:, None]) * B_mag
        delta_Z_grid = delta_Z[None, None, :, :]
    else:
        delta_Z_grid = 0.0

    det_pump = nu_grid - kvz_grid - f_grid - delta_Z_grid
    gamma_pump_ge = gamma_eff * np.sqrt(1.0 + s_pump * strength_pump)
    L_pump = 1.0 / (1.0 + 4.0 * (det_pump / gamma_pump_ge[None, None, :, :]) ** 2)
    R_pump = 0.5 * gamma_0 * s_pump * strength_pump[None, None, :, :] * L_pump

    det_probe = nu_grid + kvz_grid - f_grid - delta_Z_grid
    s_probe = kappa_probe * s_pump
    gamma_probe_ge = gamma_eff * np.sqrt(1.0 + s_probe * strength_probe)
    L_probe = 1.0 / (1.0 + 4.0 * (det_probe / gamma_probe_ge[None, None, :, :]) ** 2)
    R_probe = 0.5 * gamma_0 * s_probe * strength_probe[None, None, :, :] * L_probe

    R_total = R_pump + R_probe
    n_g_p, n_e_p = _solve_steady_state_populations(R_total, gamma_0, b_decay, gamma_tr, Ng, Ne)
    if s_probe > 1e-6:
        n_g_u, n_e_u = _solve_steady_state_populations(R_probe, gamma_0, b_decay, gamma_tr, Ng, Ne)
    else:
        n_g_u = np.ones((*nu_local_mhz.shape, *v_z.shape, Ng)) / float(Ng)
        n_e_u = np.zeros((*nu_local_mhz.shape, *v_z.shape, Ne))

    R_probe_sensor = strength_probe[None, None, :, :] * L_probe
    alpha_v_pumped = np.einsum('uvge,uvge->uv', n_g_p[:, :, :, None] - n_e_p[:, :, None, :], R_probe_sensor)
    alpha_v_unpumped = np.einsum('uvge,uvge->uv', n_g_u[:, :, :, None] - n_e_u[:, :, None, :], R_probe_sensor)

    alpha_pumped = np.trapezoid(alpha_v_pumped * f_v[None, :], v_z, axis=-1)
    alpha_unpumped = np.trapezoid(alpha_v_unpumped * f_v[None, :], v_z, axis=-1)
    diff = alpha_unpumped - alpha_pumped

    return alpha_unpumped, alpha_pumped, diff


def simulate_manifold_density_matrix(
    nu_local_mhz,
    manifold: HyperfineManifold,
    Bx_gauss,
    By_gauss,
    Bz_gauss,
    epsilon_pump,
    epsilon_probe,
    T,
    s_pump,
    kappa_probe,
    gamma_eff,
    gamma_tr,
    N_v,
):
    """
    Approach 2: Adiabatic Ground-State Density Matrix solver.
    Solves for the slow ground-state density matrix rho_g (N_sub x N_sub)
    with exact 3D Zeeman precession, Raman two-photon interference,
    optical loss, and spontaneous emission feeding.
    """
    atom_info = manifold.atom_data
    mass = atom_info.mass
    k_wave = atom_info.k_wave
    gamma_0 = atom_info.natural_gamma_mhz
    F_g_target = manifold.F_ground
    F_ref = manifold.F_ref

    Ng = atom_info.Ng
    Ne = atom_info.Ne
    dq = atom_info.dq
    b_decay = atom_info.b_decay
    ground_F = atom_info.ground_F
    ground_m = atom_info.ground_m
    excited_F = atom_info.excited_F
    E_e = atom_info.E_e
    mu_z_g = atom_info.mu_z_g
    mu_z_e = atom_info.mu_z_e

    g_target_indices = np.where(ground_F == F_g_target)[0]
    N_sub = len(g_target_indices)
    m_sub = ground_m[g_target_indices]
    mu_g_sub = mu_z_g[g_target_indices]

    mu_x_sub = atom_info.mu_x_mat_g[np.ix_(g_target_indices, g_target_indices)]
    mu_y_sub = atom_info.mu_y_mat_g[np.ix_(g_target_indices, g_target_indices)]
    mu_z_sub = atom_info.mu_z_mat_g[np.ix_(g_target_indices, g_target_indices)]

    # Full ground-state Zeeman Hamiltonian in MHz
    HZ_sub = mu_x_sub * Bx_gauss + mu_y_sub * By_gauss + mu_z_sub * Bz_gauss

    E_ref = E_e[excited_F == F_ref][0]
    f_res_local = E_e - E_ref

    E_plus_pump = np.sqrt(0.5 * (1.0 + np.sin(2.0 * epsilon_pump)))
    E_minus_pump = np.sqrt(0.5 * (1.0 - np.sin(2.0 * epsilon_pump)))

    E_plus_probe = np.sqrt(0.5 * (1.0 + np.sin(2.0 * epsilon_probe)))
    E_minus_probe = np.sqrt(0.5 * (1.0 - np.sin(2.0 * epsilon_probe)))

    dq_sub = dq[:, g_target_indices, :]
    V_pump = E_plus_pump * dq_sub[0] - E_minus_pump * dq_sub[2]
    V_probe = E_plus_probe * dq_sub[0] - E_minus_probe * dq_sub[2]

    v_th = np.sqrt(2.0 * k * T / mass)
    v_max = 350.0
    v_z = np.linspace(-v_max, v_max, N_v)
    f_v = (1.0 / (np.sqrt(np.pi) * v_th)) * np.exp(-(v_z / v_th) ** 2)
    kvz = (k_wave * v_z) / (2.0 * np.pi * 1e6)

    nu_grid = nu_local_mhz[:, None, None, None]
    kvz_grid = kvz[None, :, None, None]
    f_grid = f_res_local[None, None, None, :]

    B_mag = np.sqrt(Bx_gauss**2 + By_gauss**2 + Bz_gauss**2)
    if B_mag > 1e-6:
        delta_Z = (mu_z_e[None, :] - mu_g_sub[:, None]) * Bz_gauss
        delta_Z_grid = delta_Z[None, None, :, :]
    else:
        delta_Z_grid = 0.0

    det_pump = nu_grid - kvz_grid - f_grid - delta_Z_grid
    det_probe = nu_grid + kvz_grid - f_grid - delta_Z_grid

    L_pump = 1.0 / (1.0 + 4.0 * (det_pump / gamma_eff) ** 2)
    L_probe = 1.0 / (1.0 + 4.0 * (det_probe / gamma_eff) ** 2)

    s_probe = kappa_probe * s_pump
    L_pump_sym = 0.5 * (L_pump[:, :, :, None, :] + L_pump[:, :, None, :, :])
    VV_pump = V_pump[:, None, :] * V_pump[None, :, :]
    Gamma_p_mat = np.sum(VV_pump[None, None, :, :, :] * L_pump_sym, axis=-1) * (0.5 * gamma_0 * s_pump)

    L_probe_sym = 0.5 * (L_probe[:, :, :, None, :] + L_probe[:, :, None, :, :])
    VV_probe = V_probe[:, None, :] * V_probe[None, :, :]
    Gamma_pr_mat = np.sum(VV_probe[None, None, :, :, :] * L_probe_sym, axis=-1) * (0.5 * gamma_0 * s_probe)

    Gamma_total_mat = Gamma_p_mat + Gamma_pr_mat

    # Spontaneous decay feeding from excited states into target ground substates
    b_decay_sub = b_decay[:, g_target_indices]
    R_pump_ce = (0.5 * gamma_0 * s_pump) * (V_pump[None, None, :, :] ** 2) * L_pump
    R_probe_ce = (0.5 * gamma_0 * s_probe) * (V_probe[None, None, :, :] ** 2) * L_probe
    feed_total = np.einsum('uvce,ea->uvca', R_pump_ce + R_probe_ce, b_decay_sub)
    feed_probe = np.einsum('uvce,ea->uvca', R_probe_ce, b_decay_sub)

    dim = N_sub * N_sub
    shape_prefix = (len(nu_local_mhz), N_v)

    def solve_density_matrix(Gamma_mat, feed_mat):
        M = np.zeros((*shape_prefix, dim, dim), dtype=complex)
        b_vec = np.zeros((*shape_prefix, dim, 1), dtype=complex)

        for a in range(N_sub):
            for b in range(N_sub):
                idx_ab = a * N_sub + b
                M[:, :, idx_ab, idx_ab] += gamma_tr

                # Zeeman commutator: +i [HZ, rho]_ab = +i sum_c (HZ_ac rho_cb - rho_ac HZ_cb)
                for c in range(N_sub):
                    M[:, :, idx_ab, c * N_sub + b] += 1j * HZ_sub[a, c]
                    M[:, :, idx_ab, a * N_sub + c] -= 1j * HZ_sub[c, b]
                    M[:, :, idx_ab, c * N_sub + b] += 0.5 * Gamma_mat[:, :, a, c]
                    M[:, :, idx_ab, a * N_sub + c] += 0.5 * Gamma_mat[:, :, c, b]

                if a == b:
                    b_vec[:, :, idx_ab, 0] = gamma_tr / float(Ng)
                    for c in range(N_sub):
                        idx_cc = c * N_sub + c
                        M[:, :, idx_ab, idx_cc] -= feed_mat[:, :, c, a]

        rho_vec = np.linalg.solve(M, b_vec)[..., 0]
        return rho_vec.reshape((*shape_prefix, N_sub, N_sub))

    rho_p = solve_density_matrix(Gamma_total_mat, feed_total)
    rho_u = solve_density_matrix(Gamma_pr_mat, feed_probe)

    alpha_v_pumped = np.real(np.einsum('uvab,uvba->uv', Gamma_pr_mat, rho_p))
    alpha_v_unpumped = np.real(np.einsum('uvab,uvba->uv', Gamma_pr_mat, rho_u))

    alpha_pumped = np.trapezoid(alpha_v_pumped * f_v[None, :], v_z, axis=-1)
    alpha_unpumped = np.trapezoid(alpha_v_unpumped * f_v[None, :], v_z, axis=-1)
    diff = alpha_unpumped - alpha_pumped

    return alpha_unpumped, alpha_pumped, diff


def simulate_manifold_zeeman(
    nu_local_mhz,
    manifold: HyperfineManifold,
    Bx_gauss,
    By_gauss,
    Bz_gauss,
    epsilon_pump,
    epsilon_probe,
    T,
    s_pump,
    kappa_probe,
    gamma_eff,
    gamma_tr,
    N_v,
    model_type,
):
    """
    Computes steady-state saturated absorption in an arbitrary 3D magnetic field B = (Bx, By, Bz)
    using either Approach 2 ('density_matrix') or Approach 3 ('coherent_operator').
    """
    if model_type == "density_matrix":
        return simulate_manifold_density_matrix(
            nu_local_mhz,
            manifold,
            Bx_gauss,
            By_gauss,
            Bz_gauss,
            epsilon_pump,
            epsilon_probe,
            T,
            s_pump,
            kappa_probe,
            gamma_eff,
            gamma_tr,
            N_v,
        )
    elif model_type in ("coherent_operator", "coherent"):
        return simulate_manifold_coherent_operator(
            nu_local_mhz,
            manifold,
            Bx_gauss,
            By_gauss,
            Bz_gauss,
            epsilon_pump,
            epsilon_probe,
            T,
            s_pump,
            kappa_probe,
            gamma_eff,
            gamma_tr,
            N_v,
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Must be 'coherent_operator' or 'density_matrix'."
        )


PARAM_NAMES = [
    "fsr",
    "gamma_eff",
    "gamma_tr",
    "s_pump",
    "kappa_probe",
    "epsilon_pump",
    "epsilon_probe",
    "Bx_gauss",
    "By_gauss",
    "Bz_gauss",
    "f0_0",
    "f0_1",
    "f0_2",
    "f0_3",
]

PARAM_ALIASES = {
    "bx": "Bx_gauss",
    "bx_gauss": "Bx_gauss",
    "by": "By_gauss",
    "by_gauss": "By_gauss",
    "bz": "Bz_gauss",
    "bz_gauss": "Bz_gauss",
    "b_z": "Bz_gauss",
    "b": "B_gauss",
    "b_gauss": "B_gauss",
    "b_field": "B_gauss",
    "magnetic_field": "B_gauss",
    "eps_pump": "epsilon_pump",
    "eps_probe": "epsilon_probe",
    "g_eff": "gamma_eff",
    "g_tr": "gamma_tr",
    "s": "s_pump",
    "kappa": "kappa_probe",
}


@dataclass
class GlobalFitResult:
    """Stores the optimized parameter values from a unified multi-manifold fit."""
    fsr: float
    gamma_eff: float
    gamma_tr: float
    s_pump: float
    kappa_probe: float
    epsilon_pump: float
    epsilon_probe: float
    Bx_gauss: float
    By_gauss: float
    Bz_gauss: float
    f0: list[float]
    amp: list[float]
    bg: list[float]
    cost: float
    nfev: int
    success: bool
    x_sub: list[np.ndarray]
    y_sub: list[np.ndarray]
    fixed_params: dict[str, float]
    model_type: str

    @property
    def B_gauss(self) -> float:
        """Total magnetic field magnitude in Gauss: sqrt(Bx^2 + By^2 + Bz^2)."""
        return float(np.sqrt(self.Bx_gauss**2 + self.By_gauss**2 + self.Bz_gauss**2))


def fit_global_forward_model(
    freq_fringes,
    diff_signal,
    fringe_windows,
    p0,
    bounds_lower,
    bounds_upper,
    fixed_params,
    subsample_step,
    max_nfev,
    model_type,
):
    """
    Fits a unified forward physical model across all four rubidium D2 manifolds simultaneously.
    Fringe windows, initial parameters (p0), bounds, fixed parameters, and model_type are passed explicitly from the caller.
    """
    slices = []
    for m, (r_min, r_max) in zip(MANIFOLDS, fringe_windows):
        mask = (freq_fringes >= r_min) & (freq_fringes <= r_max)
        x_m = freq_fringes[mask]
        y_m = diff_signal[mask]
        step = max(1, len(x_m) // subsample_step)
        slices.append({
            'm': m,
            'x': x_m[::step],
            'y': y_m[::step],
        })

    # Normalize fixed parameters dict
    fixed_norm = {}
    if fixed_params:
        for k, v in fixed_params.items():
            if k == "f0" and hasattr(v, "__len__"):
                for idx, val in enumerate(v):
                    fixed_norm[f"f0_{idx}"] = float(val)
            else:
                norm_k = PARAM_ALIASES.get(k.lower(), k)
                if norm_k == "B_gauss":
                    fixed_norm["Bx_gauss"] = 0.0
                    fixed_norm["By_gauss"] = 0.0
                    fixed_norm["Bz_gauss"] = float(v)
                elif norm_k in PARAM_NAMES:
                    fixed_norm[norm_k] = float(v)

    free_indices = []
    p0_free = []
    bounds_lower_free = []
    bounds_upper_free = []
    full_fixed_values = {}

    for idx, name in enumerate(PARAM_NAMES):
        if name in fixed_norm:
            full_fixed_values[idx] = fixed_norm[name]
        else:
            free_indices.append(idx)
            p0_free.append(p0[idx])
            bounds_lower_free.append(bounds_lower[idx])
            bounds_upper_free.append(bounds_upper[idx])

    def _build_full_p(p_free):
        p_full = np.empty(len(PARAM_NAMES), dtype=float)
        for i_free, orig_idx in enumerate(free_indices):
            p_full[orig_idx] = p_free[i_free]
        for orig_idx, fix_val in full_fixed_values.items():
            p_full[orig_idx] = fix_val
        return p_full

    def residuals(p_free):
        p_full = _build_full_p(p_free)
        fsr, g_eff, g_tr, s_pump, kappa_pr, eps_pump, eps_probe = p_full[0:7]
        Bx_g, By_g, Bz_g = p_full[7:10]
        f0s = p_full[10:14]

        all_res = []
        for i, s in enumerate(slices):
            nu = (s['x'] - f0s[i]) * fsr
            _, _, diff = simulate_manifold_zeeman(
                nu,
                s['m'],
                Bx_g,
                By_g,
                Bz_g,
                eps_pump,
                eps_probe,
                295.0,
                s_pump,
                kappa_pr,
                g_eff,
                g_tr,
                81,
                model_type,
            )
            X = np.column_stack([diff, np.ones_like(diff)])
            c, _, _, _ = np.linalg.lstsq(X, s['y'], rcond=None)
            y_fit = X @ c
            all_res.append(y_fit - s['y'])
        return np.concatenate(all_res)

    if len(free_indices) > 0:
        res = least_squares(
            residuals,
            p0_free,
            bounds=(bounds_lower_free, bounds_upper_free),
            ftol=1e-3,
            xtol=1e-3,
            max_nfev=max_nfev,
            verbose=0,
        )
        p_opt = _build_full_p(res.x)
        cost = float(res.cost)
        nfev = int(res.nfev)
        success = bool(res.success)
    else:
        p_opt = _build_full_p([])
        res_arr = residuals([])
        cost = float(0.5 * np.sum(res_arr ** 2))
        nfev = 1
        success = True

    fsr, g_eff, g_tr, s_pump, kappa_pr, eps_pump, eps_probe = p_opt[0:7]
    Bx_g, By_g, Bz_g = p_opt[7:10]
    f0s = p_opt[10:14]

    amps, bgs = [], []
    for i, s in enumerate(slices):
        nu = (s['x'] - f0s[i]) * fsr
        _, _, diff = simulate_manifold_zeeman(
            nu,
            s['m'],
            Bx_g,
            By_g,
            Bz_g,
            eps_pump,
            eps_probe,
            295.0,
            s_pump,
            kappa_pr,
            g_eff,
            g_tr,
            81,
            model_type,
        )
        X = np.column_stack([diff, np.ones_like(diff)])
        c, _, _, _ = np.linalg.lstsq(X, s['y'], rcond=None)
        amps.append(float(c[0]))
        bgs.append(float(c[1]))

    return GlobalFitResult(
        fsr=float(fsr),
        gamma_eff=float(g_eff),
        gamma_tr=float(g_tr),
        s_pump=float(s_pump),
        kappa_probe=float(kappa_pr),
        epsilon_pump=float(eps_pump),
        epsilon_probe=float(eps_probe),
        Bx_gauss=float(Bx_g),
        By_gauss=float(By_g),
        Bz_gauss=float(Bz_g),
        f0=[float(x) for x in f0s],
        amp=amps,
        bg=bgs,
        cost=cost,
        nfev=nfev,
        success=success,
        x_sub=[s['x'] for s in slices],
        y_sub=[s['y'] for s in slices],
        fixed_params=fixed_norm,
        model_type=model_type,
    )
