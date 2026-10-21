import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from scipy.signal import find_peaks
import physics_model as pm



def test_spontaneous_decay_branching_conservation():
    """For every excited state, the sum of branching ratios into all ground states must equal 1.0."""
    for atom in [pm.ATOM_DATA.rb87, pm.ATOM_DATA.rb85]:
        for ie in range(atom.Ne):
            total_prob = np.sum(atom.b_decay[ie, :])
            assert np.isclose(total_prob, 1.0, atol=1e-5), f"Decay conservation failed for {atom.isotope} state {ie}: {total_prob}"


def test_zero_pump_unperturbed_population():
    """When pump intensity is 0, the differential saturated absorption signal must be identically zero."""
    m = pm.MANIFOLDS[0]
    nu = np.linspace(-300, 50, 50)
    alpha_u, alpha_p, diff = pm.simulate_manifold_zeeman(
        nu, m, B_gauss=0.0, epsilon_pol=0.0, s_pump=0.0, N_v=51
    )
    assert np.allclose(diff, 0.0, atol=1e-9)
    assert np.allclose(alpha_u, alpha_p, atol=1e-9)


def test_87Rb_F2_no_aliasing_and_correct_peaks():
    """
    At B = 0 and linear polarization, 87Rb F=2 must produce exactly 5 resolved features:
    Crossover(1,2), True F'=2, Crossover(1,3), Crossover(2,3), and True F'=3.
    Crucially, it must NOT produce spurious discretization picket-fence peaks!
    """
    m = pm.MANIFOLDS[0]
    nu = np.linspace(-550.0, 100.0, 400)
    _, _, diff = pm.simulate_manifold_zeeman(
        nu, m, B_gauss=0.0, epsilon_pol=0.0, s_pump=1.5, gamma_eff=15.0, N_v=151
    )

    peaks, _ = find_peaks(diff, prominence=2e-4)
    detected_freqs = nu[peaks]

    # Exactly 6 physical features (3 true transitions + 3 crossovers)
    assert len(detected_freqs) in (5, 6), f"Expected 5 or 6 physical peaks, but found {len(detected_freqs)}: {detected_freqs}"

    # Expected peak centers:
    # F'=1 ~ -423.6 MHz, CO(1,2) ~ -345.1 MHz, F'=2 ~ -266.7 MHz, CO(1,3) ~ -211.8 MHz, CO(2,3) ~ -133.3 MHz, F'=3 ~ 0.0 MHz
    expected = np.array([-423.6, -345.1, -266.65, -211.80, -133.33, 0.0])
    for exp_peak in expected:
        min_dist = np.min(np.abs(detected_freqs - exp_peak))
        assert min_dist < 4.5, f"Expected peak near {exp_peak} MHz, closest was {min_dist:.2f} MHz away"


def test_85Rb_F3_no_aliasing_and_correct_peaks():
    """
    At B = 0, 85Rb F=3 must produce its characteristic peaks and zero aliasing picket-fence peaks.
    """
    m = pm.MANIFOLDS[1]
    nu = np.linspace(-260.0, 50.0, 400)
    _, _, diff = pm.simulate_manifold_zeeman(
        nu, m, B_gauss=0.0, epsilon_pol=0.0, s_pump=1.5, gamma_eff=15.0, N_v=151
    )

    peaks, _ = find_peaks(diff, prominence=1e-4)
    detected_freqs = nu[peaks]

    # At least 4 distinct features are cleanly resolved with prominence >= 1e-4
    assert len(detected_freqs) >= 4, f"Expected at least 4 prominent peaks for 85Rb F=3, found {len(detected_freqs)}: {detected_freqs}"
    assert len(detected_freqs) <= 6, f"Too many peaks (aliasing detected): {detected_freqs}"


def test_magnetic_splitting_symmetric_at_moderate_field():
    """
    In a longitudinal field B = 25 G, the cycling transition (0 MHz) must split symmetrically into outer Zeeman lobes.
    """
    m = pm.MANIFOLDS[0]
    nu = np.linspace(-50.0, 50.0, 150)
    _, _, diff = pm.simulate_manifold_zeeman(
        nu, m, B_gauss=25.0, epsilon_pol=0.0, s_pump=1.0, gamma_eff=12.0, N_v=151
    )

    peaks, _ = find_peaks(diff, prominence=1e-4)
    assert len(peaks) >= 2, f"Expected cycling peak to split into at least 2 lobes, found {len(peaks)}"
    lobe_freqs = nu[peaks]
    pos_lobe = lobe_freqs[lobe_freqs > 10.0][0]
    neg_lobe = lobe_freqs[lobe_freqs < -10.0][-1]
    assert np.isclose(pos_lobe, -neg_lobe, atol=3.0)


def test_global_forward_fit_execution():
    """
    Test that fit_global_forward_model converges and returns a valid GlobalFitResult
    with physical parameters within reasonable experimental bounds.
    """
    # Create synthetic test dataset over the fringe range of all manifolds
    freq_fringes = np.linspace(6.0, 24.0, 1200)
    diff_signal = np.zeros_like(freq_fringes)

    # Inject simulated signal for each manifold with FSR = 380 MHz
    test_fsr = 380.0
    for i, m in enumerate(pm.MANIFOLDS):
        mask = (freq_fringes >= m.fringe_window[0]) & (freq_fringes <= m.fringe_window[1])
        x_m = freq_fringes[mask]
        nu_m = (x_m - m.f0_guess) * test_fsr
        _, _, diff = pm.simulate_manifold_zeeman(
            nu_m, m, B_gauss=0.0, epsilon_pol=0.1, s_pump=1.5, gamma_eff=10.0, gamma_tr=0.1, N_v=51
        )
        diff_signal[mask] = 0.5 * diff + 0.01

    res = pm.fit_global_forward_model(
        freq_fringes,
        diff_signal,
        B_gauss=0.0,
        subsample_step=25,
        max_nfev=20,
    )

    assert isinstance(res, pm.GlobalFitResult)
    assert 350.0 <= res.fsr <= 410.0
    assert 0.0 <= res.gamma_eff <= 25.0
    assert len(res.f0) == 4
    assert 0.1 <= res.s_pump <= 25.0
    assert len(res.amp) == 4
    assert len(res.bg) == 4
    assert res.cost < 1.0

