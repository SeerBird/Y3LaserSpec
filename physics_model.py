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
    dq: np.ndarray
    b_decay: np.ndarray

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
        dq=dq,
        b_decay=b_decay,
    )


def _build_atom_structures() -> AtomRegistry:
    return AtomRegistry(
        rb87=_build_single_atom_structure('87Rb', '87'),
        rb85=_build_single_atom_structure('85Rb', '85'),
    )


ATOM_DATA = _build_atom_structures()


@dataclass
class HyperfineManifold:
    """Represents a specific hyperfine ground-state absorption manifold."""
    name: str
    isotope: str
    F_ground: int
    F_ref: int
    fringe_window: tuple[float, float]
    f0_guess: float

    @property
    def atom_data(self) -> AtomicStructure:
        return ATOM_DATA[self.isotope]

    def __getitem__(self, item):
        return getattr(self, item)


# Hyperfine manifolds under study
MANIFOLDS = [
    HyperfineManifold(
        name='87Rb F=2',
        isotope='87',
        F_ground=2,
        F_ref=3,
        fringe_window=(9.0, 10.6),
        f0_guess=10.33,
    ),
    HyperfineManifold(
        name='85Rb F=3',
        isotope='85',
        F_ground=3,
        F_ref=4,
        fringe_window=(12.6, 13.6),
        f0_guess=13.34,
    ),
    HyperfineManifold(
        name='85Rb F=2',
        isotope='85',
        F_ground=2,
        F_ref=1,
        fringe_window=(20.6, 21.4),
        f0_guess=20.88,
    ),
    HyperfineManifold(
        name='87Rb F=1',
        isotope='87',
        F_ground=1,
        F_ref=0,
        fringe_window=(27.1, 28.1),
        f0_guess=27.28,
    ),
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


def simulate_manifold_zeeman(
    nu_local_mhz,
    manifold: HyperfineManifold,
    B_gauss,
    epsilon_pump,
    epsilon_probe,
    delta_theta,
    T,
    s_pump,
    kappa_probe,
    gamma_eff,
    gamma_tr,
    N_v,
):
    """
    Computes steady-state saturated absorption with full Zeeman sublevel optical pumping
    in a longitudinal magnetic field (k parallel to B).
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
    excited_F = atom_info.excited_F
    E_e = atom_info.E_e
    mu_z_g = atom_info.mu_z_g
    mu_z_e = atom_info.mu_z_e

    # Reference frequency for the manifold (e.g. cycling line)
    E_ref = E_e[excited_F == F_ref][0]
    f_res_local = E_e - E_ref

    # Polarization fractions for transverse light propagating along quantization axis (k || B).
    # pylcp dq ordering: index 0 is delta_m = +1 (sigma+), 1 is pi, 2 is delta_m = -1 (sigma-).
    p_plus_pump = (1.0 + np.sin(2.0 * epsilon_pump)) / 2.0
    p_minus_pump = (1.0 - np.sin(2.0 * epsilon_pump)) / 2.0
    p_q_pump = np.array([p_plus_pump, 0.0, p_minus_pump])

    # Probe beam polarization (independent parameters):
    p_plus_probe = (1.0 + np.sin(2.0 * epsilon_probe)) / 2.0
    p_minus_probe = (1.0 - np.sin(2.0 * epsilon_probe)) / 2.0
    p_q_probe = np.array([p_plus_probe, 0.0, p_minus_probe])

    # Resonant velocity window: atoms outside this window never hit pump & probe simultaneously
    v_th = np.sqrt(2.0 * k * T / mass)
    v_max = 350.0  # m/s (covers > 450 MHz Doppler shift)
    v_z = np.linspace(-v_max, v_max, N_v)
    f_v = (1.0 / (np.sqrt(np.pi) * v_th)) * np.exp(-(v_z / v_th) ** 2)
    kvz = (k_wave * v_z) / (2.0 * np.pi * 1e6)  # MHz

    # Broadcasting shapes: (N_nu, 1, 1, 1), (1, N_v, 1, 1), (1, 1, 1, Ne), (1, 1, Ng, Ne)
    nu_grid = nu_local_mhz[:, None, None, None]
    kvz_grid = kvz[None, :, None, None]
    f_grid = f_res_local[None, None, None, :]

    # Zeeman shifts: Delta nu_Z = Delta E_e - Delta E_g = (mu_z_e - mu_z_g) * B
    if abs(B_gauss) > 1e-6:
        delta_Z = (mu_z_e[None, :] - mu_z_g[:, None]) * B_gauss
        delta_Z_grid = delta_Z[None, None, :, :]
    else:
        delta_Z_grid = 0.0

    # Mask for transitions belonging to this manifold
    target_mask = (ground_F == F_g_target)

    # 1. PUMP TRANSITIONS (+z)
    det_pump = nu_grid - kvz_grid - f_grid - delta_Z_grid
    strength_pump = np.sum(p_q_pump[:, None, None] * (dq ** 2), axis=0) * target_mask[:, None]
    gamma_pump_ge = gamma_eff * np.sqrt(1.0 + s_pump * strength_pump)
    L_pump = 1.0 / (1.0 + 4.0 * (det_pump / gamma_pump_ge[None, None, :, :]) ** 2)
    R_pump = 0.5 * gamma_0 * s_pump * strength_pump[None, None, :, :] * L_pump

    # 2. PROBE TRANSITIONS (-z)
    det_probe = nu_grid + kvz_grid - f_grid - delta_Z_grid
    strength_probe = np.sum(p_q_probe[:, None, None] * (dq ** 2), axis=0) * target_mask[:, None]
    s_probe = kappa_probe * s_pump
    gamma_probe_ge = gamma_eff * np.sqrt(1.0 + s_probe * strength_probe)
    L_probe = 1.0 / (1.0 + 4.0 * (det_probe / gamma_probe_ge[None, None, :, :]) ** 2)
    R_probe = 0.5 * gamma_0 * s_probe * strength_probe[None, None, :, :] * L_probe

    # 3. Steady-state populations for pumped measurement (both pump & probe present)
    R_total = R_pump + R_probe
    n_g_p, n_e_p = _solve_steady_state_populations(R_total, gamma_0, b_decay, gamma_tr, Ng, Ne)

    # 4. Steady-state populations for unpumped reference measurement (probe alone)
    if s_probe > 1e-6:
        n_g_u, n_e_u = _solve_steady_state_populations(R_probe, gamma_0, b_decay, gamma_tr, Ng, Ne)
    else:
        n_g_u = np.ones((*nu_local_mhz.shape, *v_z.shape, Ng)) / float(Ng)
        n_e_u = np.zeros((*nu_local_mhz.shape, *v_z.shape, Ne))

    # 5. Readout probe absorption (accounting for absorption and stimulated emission)
    R_probe_sensor = strength_probe[None, None, :, :] * L_probe
    alpha_v_pumped = np.einsum('uvge,uvge->uv', n_g_p[:, :, :, None] - n_e_p[:, :, None, :], R_probe_sensor)
    alpha_v_unpumped = np.einsum('uvge,uvge->uv', n_g_u[:, :, :, None] - n_e_u[:, :, None, :], R_probe_sensor)
    diff_v = alpha_v_unpumped - alpha_v_pumped

    # 6. 1D velocity integral over Maxwell-Boltzmann distribution
    alpha_pumped = np.trapezoid(alpha_v_pumped * f_v[None, :], v_z, axis=-1)
    alpha_unpumped = np.trapezoid(alpha_v_unpumped * f_v[None, :], v_z, axis=-1)
    
    # 7. Polarization mode overlap and Larmor precession dephasing
    P_geom = (np.cos(delta_theta)**2 * np.cos(epsilon_pump - epsilon_probe)**2 + 
              np.sin(delta_theta)**2 * np.sin(epsilon_pump + epsilon_probe)**2)
    P_circ = 0.5 * (1.0 + np.sin(2.0 * epsilon_pump) * np.sin(2.0 * epsilon_probe))
    omega_L = 4.4 * abs(B_gauss)
    dephasing = 1.0 / (1.0 + (2.0 * omega_L / gamma_tr)**2) if gamma_tr > 0 else 1.0
    P_eff = float(np.clip(P_circ + (P_geom - P_circ) * dephasing, 1e-4, 1.0))

    diff = P_eff * (alpha_unpumped - alpha_pumped)

    return alpha_unpumped, alpha_pumped, diff


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
    delta_theta: float
    f0: list[float]
    amp: list[float]
    bg: list[float]
    cost: float
    nfev: int
    success: bool
    x_sub: list[np.ndarray]
    y_sub: list[np.ndarray]


def fit_global_forward_model(
    freq_fringes,
    diff_signal,
    p0,
    bounds_lower,
    bounds_upper,
    B_gauss,
    subsample_step,
    max_nfev,
):
    """
    Fits a unified forward physical model across all four rubidium D2 manifolds simultaneously.
    Initial parameters (p0) and bounds are passed explicitly from the caller.
    """
    slices = []
    for m in MANIFOLDS:
        r_min, r_max = m.fringe_window
        mask = (freq_fringes >= r_min) & (freq_fringes <= r_max)
        x_m = freq_fringes[mask]
        y_m = diff_signal[mask]
        step = max(1, len(x_m) // subsample_step)
        slices.append({
            'm': m,
            'x': x_m[::step],
            'y': y_m[::step],
        })

    def residuals(p):
        fsr, g_eff, g_tr, s_pump, kappa_pr, eps_pump, eps_probe, delta_th = p[0:8]
        f0s = p[8:12]

        all_res = []
        for i, s in enumerate(slices):
            nu = (s['x'] - f0s[i]) * fsr
            _, _, diff = simulate_manifold_zeeman(
                nu,
                s['m'],
                B_gauss,
                eps_pump,
                eps_probe,
                delta_th,
                295.0,
                s_pump,
                kappa_pr,
                g_eff,
                g_tr,
                81,
            )
            X = np.column_stack([diff, np.ones_like(diff)])
            c, _, _, _ = np.linalg.lstsq(X, s['y'], rcond=None)
            y_fit = X @ c
            all_res.append(y_fit - s['y'])
        return np.concatenate(all_res)

    res = least_squares(
        residuals,
        p0,
        bounds=(bounds_lower, bounds_upper),
        ftol=1e-3,
        xtol=1e-3,
        max_nfev=max_nfev,
        verbose=0,
    )

    fsr, g_eff, g_tr, s_pump, kappa_pr, eps_pump, eps_probe, delta_th = res.x[0:8]
    f0s = res.x[8:12]

    amps, bgs = [], []
    for i, s in enumerate(slices):
        nu = (s['x'] - f0s[i]) * fsr
        _, _, diff = simulate_manifold_zeeman(
            nu,
            s['m'],
            B_gauss,
            eps_pump,
            eps_probe,
            delta_th,
            295.0,
            s_pump,
            kappa_pr,
            g_eff,
            g_tr,
            81,
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
        delta_theta=float(delta_th),
        f0=[float(x) for x in f0s],
        amp=amps,
        bg=bgs,
        cost=float(res.cost),
        nfev=int(res.nfev),
        success=bool(res.success),
        x_sub=[s['x'] for s in slices],
        y_sub=[s['y'] for s in slices],
    )
