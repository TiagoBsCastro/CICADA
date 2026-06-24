import numpy as np
import pytest
from cicada.power import power_spectrum_3d
from cicada.density import density_grid_3d

def test_power_spectrum_white_noise():
    """White noise on the grid should recover P(k) = V / N_cells * sigma^2."""
    np.random.seed(42)
    N = 32
    BoxSize = 100.0
    sigma = 2.0

    delta = np.random.normal(0, sigma, (N, N, N))

    k_3d, pk_3d, nmodes_3d = power_spectrum_3d(delta, boxsize=BoxSize, scheme="None")

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

    _, pk_none_3d, _ = power_spectrum_3d(delta, boxsize=BoxSize, scheme="None")
    _, pk_ngp_3d, _  = power_spectrum_3d(delta, boxsize=BoxSize, scheme="NGP")
    _, pk_cic_3d, _  = power_spectrum_3d(delta, boxsize=BoxSize, scheme="CIC")

    assert pk_cic_3d[-1] > pk_ngp_3d[-1]
    assert pk_ngp_3d[-1] > pk_none_3d[-1]

def test_power_spectrum_returns():
    """Returned arrays should be 1D and same length."""
    delta = np.zeros((16, 16, 16))
    k_3d, pk_3d, nmodes_3d = power_spectrum_3d(delta, boxsize=50.0, scheme="None")

    assert k_3d.ndim == 1
    assert pk_3d.ndim == 1
    assert nmodes_3d.ndim == 1
    assert len(k_3d) == len(pk_3d) == len(nmodes_3d)

def test_interlacing_requires_tuple():
    """interlacing=True with a single array should raise ValueError."""
    delta = np.zeros((16, 16, 16))
    with pytest.raises(ValueError, match="tuple"):
        power_spectrum_3d(delta, boxsize=50.0, interlacing=True)

def test_interlacing_suppresses_aliasing():
    """
    Interlacing should make the CIC power spectrum closer to the uncorrected
    white-noise truth at high k, compared to the non-interlaced version.

    We paint white-noise particles onto a grid with CIC, compute P(k) with and
    without interlacing, and check that interlacing reduces the scatter at high k.
    """
    np.random.seed(123)
    N = 32
    BoxSize = 100.0
    grid = N
    cell_size = BoxSize / grid
    N_particles = 50000

    pos = np.random.uniform(0, BoxSize, (N_particles, 3))

    # Original density field
    delta_orig = density_grid_3d(
        pos, boxsize=BoxSize, grid=grid, scheme="CIC",
        periodic=True, overdensity=True,
    )

    # Shifted density field (half-cell shift in each dimension)
    pos_shifted = np.mod(pos + 0.5 * cell_size, BoxSize)
    delta_shifted = density_grid_3d(
        pos_shifted, boxsize=BoxSize, grid=grid, scheme="CIC",
        periodic=True, overdensity=True,
    )

    # P(k) without interlacing
    k_no, pk_no, _ = power_spectrum_3d(
        delta_orig, boxsize=BoxSize, scheme="CIC",
    )

    # P(k) with interlacing
    k_il, pk_il, _ = power_spectrum_3d(
        (delta_orig, delta_shifted), boxsize=BoxSize, scheme="CIC",
        interlacing=True,
    )

    # For uniform random particles the true P(k) is flat (shot noise).
    # The theoretical shot-noise level is V / N_particles.
    P_shot = BoxSize**3 / N_particles

    # At high k the non-interlaced spectrum deviates more from P_shot due to
    # aliasing than the interlaced one.  Compare the last quarter of k bins.
    n_high = max(1, len(k_no) // 4)

    residual_no = np.abs(pk_no[-n_high:] / P_shot - 1.0)
    residual_il = np.abs(pk_il[-n_high:] / P_shot - 1.0)

    # Interlaced residuals should be smaller on average
    assert np.mean(residual_il) < np.mean(residual_no)
