import numpy as np
import pytest
from cicada.power import power_spectrum_3d


def _cosine_mode(N, mode, shift_cells=(0.0, 0.0, 0.0)):
    """Return a periodic cosine mode sampled on an N^3 grid."""
    x, y, z = np.indices((N, N, N), dtype=float)
    arg = (
        2.0
        * np.pi
        / N
        * (
            mode[0] * (x - shift_cells[0])
            + mode[1] * (y - shift_cells[1])
            + mode[2] * (z - shift_cells[2])
        )
    )
    return np.cos(arg)


def test_power_spectrum_white_noise():
    """White noise on the grid should recover P(k) = V / N_cells * sigma^2."""
    np.random.seed(42)
    N = 32
    BoxSize = 100.0
    sigma = 2.0

    delta = np.random.normal(0, sigma, (N, N, N))
    _, pk_3d, _ = power_spectrum_3d(
        delta, boxsize=BoxSize, scheme="None", kmax="grid"
    )

    V = BoxSize**3
    N_cells = N**3
    P_theory = (V / N_cells) * (sigma**2)

    assert np.isclose(np.mean(pk_3d), P_theory, rtol=0.1)


def test_power_spectrum_mas_correction():
    """MAS corrections should boost high-k power: CIC > NGP > None."""
    np.random.seed(42)
    N = 32
    BoxSize = 100.0
    delta = np.random.normal(0, 1.0, (N, N, N))

    _, pk_none_3d, _ = power_spectrum_3d(
        delta, boxsize=BoxSize, scheme="None", kmax="grid"
    )
    _, pk_ngp_3d, _ = power_spectrum_3d(
        delta, boxsize=BoxSize, scheme="NGP", kmax="grid"
    )
    _, pk_cic_3d, _ = power_spectrum_3d(
        delta, boxsize=BoxSize, scheme="CIC", kmax="grid"
    )

    assert pk_cic_3d[-1] > pk_ngp_3d[-1]
    assert pk_ngp_3d[-1] > pk_none_3d[-1]


def test_power_spectrum_returns():
    """Returned arrays should be 1D and same length."""
    delta = np.zeros((16, 16, 16))
    k_3d, pk_3d, nmodes_3d = power_spectrum_3d(
        delta, boxsize=50.0, scheme="None"
    )

    assert k_3d.ndim == 1
    assert pk_3d.ndim == 1
    assert nmodes_3d.ndim == 1
    assert len(k_3d) == len(pk_3d) == len(nmodes_3d)


def test_power_spectrum_default_kmax_is_scalar_nyquist():
    """The default kmax should not include corner modes beyond scalar Nyquist."""
    np.random.seed(1)
    N = 32
    BoxSize = 100.0
    delta = np.random.normal(0, 1.0, (N, N, N))

    k_default, _, _ = power_spectrum_3d(delta, boxsize=BoxSize, scheme="None")
    k_grid, _, _ = power_spectrum_3d(
        delta, boxsize=BoxSize, scheme="None", kmax="grid"
    )

    kf = 2.0 * np.pi / BoxSize
    k_nyquist = (N // 2) * kf

    assert k_default[-1] <= k_nyquist
    assert k_grid[-1] > k_nyquist


def test_power_spectrum_scaled_nyquist_kmax():
    """String specifications such as '0.5*nyquist' should be accepted."""
    np.random.seed(2)
    N = 32
    BoxSize = 100.0
    delta = np.random.normal(0, 1.0, (N, N, N))

    k_half, _, _ = power_spectrum_3d(
        delta, boxsize=BoxSize, scheme="None", kmax="0.5*nyquist"
    )

    kf = 2.0 * np.pi / BoxSize
    k_half_nyquist = 0.5 * (N // 2) * kf

    assert k_half[-1] <= k_half_nyquist


def test_power_spectrum_rejects_invalid_kmax():
    """Invalid kmax values should fail loudly."""
    delta = np.zeros((16, 16, 16))
    with pytest.raises(ValueError, match="positive finite"):
        power_spectrum_3d(delta, boxsize=50.0, scheme="None", kmax=0.0)


def test_interlacing_requires_tuple():
    """interlacing=True with a single array should raise ValueError."""
    delta = np.zeros((16, 16, 16))
    with pytest.raises(ValueError, match="tuple"):
        power_spectrum_3d(delta, boxsize=50.0, interlacing=True)


def test_interlacing_recovers_half_cell_shifted_fourier_mode():
    """
    Interlacing should recover the same Fourier power for a known mode and
    its analytically half-cell shifted counterpart.

    This test directly validates the sign of the Fourier phase correction.
    """
    N = 32
    BoxSize = 100.0
    mode = (2, 3, 4)

    delta = _cosine_mode(N, mode)
    delta_shifted = _cosine_mode(N, mode, shift_cells=(0.5, 0.5, 0.5))

    k_ref, pk_ref, _ = power_spectrum_3d(
        delta, boxsize=BoxSize, scheme="None", kmax="grid"
    )
    k_interlaced, pk_interlaced, _ = power_spectrum_3d(
        (delta, delta_shifted),
        boxsize=BoxSize,
        scheme="None",
        interlacing=True,
        kmax="grid",
    )

    np.testing.assert_allclose(k_interlaced, k_ref)
    np.testing.assert_allclose(pk_interlaced, pk_ref, rtol=1.0e-12, atol=1.0e-12)


def test_interlacing_phase_test_is_sensitive_to_shift_sign():
    """The deterministic phase test should fail for the opposite shift sign."""
    N = 32
    BoxSize = 100.0
    mode = (2, 3, 4)

    delta = _cosine_mode(N, mode)
    delta_shifted_wrong = _cosine_mode(N, mode, shift_cells=(-0.5, -0.5, -0.5))

    _, pk_ref, _ = power_spectrum_3d(
        delta, boxsize=BoxSize, scheme="None", kmax="grid"
    )
    _, pk_wrong, _ = power_spectrum_3d(
        (delta, delta_shifted_wrong),
        boxsize=BoxSize,
        scheme="None",
        interlacing=True,
        kmax="grid",
    )

    mode_bin = np.argmax(pk_ref)
    assert np.abs(pk_wrong[mode_bin] - pk_ref[mode_bin]) > 0.1 * pk_ref[mode_bin]
