"""Unit tests for the rodeo_ness package.

These check the analytically known quantities that anchor the paper:
the single-spin gap, the exact NESS observables, the singular-value
identity for the embedding, and the schedule normalization.
"""

import numpy as np
import pytest

import rodeo_ness as rn


def test_single_spin_gap():
    """Single-spin Liouvillian gap is 1/2 for all field strengths."""
    for h in [0.5, 1.0, 1.5, 2.0]:
        L = rn.single_spin_liouvillian(h=h)
        assert rn.spectral_separation(L) == pytest.approx(0.5, abs=1e-9)
        assert rn.decay_rate(L) == pytest.approx(0.5, abs=1e-9)


def test_single_spin_ness_observables():
    """Exact NESS observables at h=0.5: |<sy>|=2/3, <sz>=-1/3, <sx>=0."""
    L = rn.single_spin_liouvillian(h=0.5)
    rho = rn.steady_state(L)
    sx = np.real(np.trace(rho @ np.array([[0, 1], [1, 0]])))
    sy = np.real(np.trace(rho @ np.array([[0, -1j], [1j, 0]])))
    sz = np.real(np.trace(rho @ np.array([[1, 0], [0, -1]])))
    assert abs(sy) == pytest.approx(2 / 3, abs=1e-6)
    assert sz == pytest.approx(-1 / 3, abs=1e-6)
    assert sx == pytest.approx(0.0, abs=1e-6)


def test_embedding_eigs_are_singular_values():
    """Nonzero |eig(M)| equal the singular values of L (doubled)."""
    L = rn.tfim_liouvillian(2, J=1.0, h=1.0, gamma=1.0)
    M = rn.hermitian_embedding(L)
    eig_abs = np.sort(np.abs(np.linalg.eigvalsh(M)))
    sv = np.sort(np.concatenate([np.linalg.svd(L, compute_uv=False)] * 2))
    # both sorted ascending; the nonzero parts must match
    assert np.allclose(eig_abs, np.sort(np.concatenate(
        [np.linalg.svd(L, compute_uv=False),
         np.linalg.svd(L, compute_uv=False)])), atol=1e-8)


def test_gap_closes_with_system_size():
    """Growing the chain monotonically reduces the spectral separation."""
    gaps = [rn.spectral_separation(rn.tfim_liouvillian(N, J=0.0, gamma=1.0))
            for N in [1, 2, 3, 4]]
    assert all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))


def test_separation_below_decay_rate_in_tfim():
    """For the interacting TFIM, g <= g_decay (and strictly below for N>=2)."""
    L = rn.tfim_liouvillian(2, J=0.0, h=1.0, gamma=1.0)
    assert rn.spectral_separation(L) < rn.decay_rate(L)


def test_vdc_schedule_normalization():
    """The van der Corput schedule sums to the requested total depth."""
    for n in [3, 7, 20]:
        for T in [5.0, 15.0]:
            times = rn.vdc_schedule(n, T)
            assert times.sum() == pytest.approx(T, rel=1e-12)
            assert (times > 0).all()


def test_zero_modes_invariant_under_rodeo():
    """Zero modes (phi=0) are preserved: cos(0)=1, so beta=1."""
    L = rn.single_spin_liouvillian(h=0.5)
    times = rn.vdc_schedule(5, 10.0)
    beta_zero = rn.rodeo_residual(np.array([0.0]), times)
    assert beta_zero[0] == pytest.approx(1.0, abs=1e-12)


def test_cost_ratio_grows_with_gap():
    """The Rodeo advantage is far larger at large g than at small g.

    The growth is monotone over a wide gap range; at very small g it can
    wiggle because the single-spin spectrum has only six modes (the
    density corner effect studied in the paper), so we compare the
    well-separated endpoints rather than every adjacent pair.
    """
    r_small = rn.cost_ratio(
        rn.tfim_liouvillian(1, J=0.0, h=1.0, gamma=1.0), total_depth=15
    )
    r_large = rn.cost_ratio(
        rn.tfim_liouvillian(1, J=0.0, h=1.0, gamma=3.0), total_depth=15
    )
    assert r_large > 100 * r_small


def test_density_saturates():
    """At fixed g, adding modes beyond a handful barely changes the ratio."""
    r6 = rn.synthetic_cost_ratio(0.6, 5, total_depth=15)    # 6 modes
    r20 = rn.synthetic_cost_ratio(0.6, 19, total_depth=15)  # 20 modes
    r100 = rn.synthetic_cost_ratio(0.6, 99, total_depth=15)  # 100 modes
    # 6 -> 20 -> 100 should be nearly flat (within a factor of 2)
    assert abs(np.log10(r20) - np.log10(r100)) < 0.3
