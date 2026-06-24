import numpy as np
import pytest
from cicada.power import power_spectrum_3d

def test_power_spectrum_white_noise():
    # White noise has a flat power spectrum P(k) = V / N^3 * sigma^2
    np.random.seed(42)
    N = 32
    BoxSize = 100.0
    sigma = 2.0
    
    # Generate white noise on the grid
    delta = np.random.normal(0, sigma, (N, N, N))
    
    # Compute Pk without MAS correction
    k_3d, pk_3d, nmodes_3d = power_spectrum_3d(delta, boxsize=BoxSize, scheme="None")
    
    # Theoretical amplitude
    V = BoxSize**3
    N_cells = N**3
    P_theory = (V / N_cells) * (sigma**2)
    
    # The measured P(k) should be close to P_theory on average
    assert np.isclose(np.mean(pk_3d), P_theory, rtol=0.1)

def test_power_spectrum_mas_correction():
    np.random.seed(42)
    N = 32
    BoxSize = 100.0
    delta = np.random.normal(0, 1.0, (N, N, N))
    
    _, pk_none_3d, _ = power_spectrum_3d(delta, boxsize=BoxSize, scheme="None")
    _, pk_ngp_3d, _  = power_spectrum_3d(delta, boxsize=BoxSize, scheme="NGP")
    _, pk_cic_3d, _  = power_spectrum_3d(delta, boxsize=BoxSize, scheme="CIC")
    
    # MAS corrections divide by W^2 where W < 1 for k > 0.
    # Therefore, Pk with scheme="CIC" should be > Pk with scheme="NGP" > Pk with scheme="None" at high k.
    
    # Check the last bin (highest k)
    assert pk_cic_3d[-1] > pk_ngp_3d[-1]
    assert pk_ngp_3d[-1] > pk_none_3d[-1]
    
def test_power_spectrum_returns():
    delta = np.zeros((16, 16, 16))
    k_3d, pk_3d, nmodes_3d = power_spectrum_3d(delta, boxsize=50.0, scheme="None")
    
    assert k_3d.ndim == 1
    assert pk_3d.ndim == 1
    assert nmodes_3d.ndim == 1
    assert len(k_3d) == len(pk_3d) == len(nmodes_3d)
