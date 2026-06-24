import numpy as np

def power_spectrum_3d(delta, boxsize, scheme="CIC"):
    """
    Compute the auto-power spectrum of a 3D density field.
    
    Parameters
    ----------
    delta : array_like
        3D density contrast field.
    boxsize : float
        Size of the simulation box.
    scheme : str, optional
        Mass-assignment scheme used to generate the density field 
        ('NGP', 'CIC', 'TSC', 'PCS', 'None'). Default is 'CIC'.
        
    Returns
    -------
    k_3d : ndarray
        1D array of wavenumber bin centers.
    pk_3d : ndarray
        1D array of the monopole power spectrum P(k).
    nmodes_3d : ndarray
        1D array containing the number of modes in each k bin.
    """
    delta = np.asarray(delta)
    if delta.ndim != 3:
        raise ValueError("delta must be a 3D array")
        
    N = delta.shape[0]
    if delta.shape != (N, N, N):
        raise ValueError("delta must be a cubic grid")
        
    boxsize = float(boxsize)
    kf = 2.0 * np.pi / boxsize
    
    # Compute FFT
    delta_k = np.fft.rfftn(delta)
    
    # Calculate frequencies
    k_x = np.fft.fftfreq(N) * N * kf
    k_y = np.fft.fftfreq(N) * N * kf
    k_z = np.fft.rfftfreq(N) * N * kf
    
    Kx, Ky, Kz = np.meshgrid(k_x, k_y, k_z, indexing='ij')
    K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)
    
    # Power spectrum amplitude
    # numpy forward FFT does not divide by N^3, so we divide the squared amplitude by N^6.
    # We multiply by boxsize**3 (the volume) to get the correct dimensions for P(k).
    P_k_3d = (boxsize**3 / (N**6)) * np.abs(delta_k)**2
    
    # MAS window correction
    if scheme is not None and str(scheme).upper() != "NONE":
        scheme_upper = str(scheme).upper()
        if scheme_upper == "NGP":
            p = 1
        elif scheme_upper == "CIC":
            p = 2
        elif scheme_upper == "TSC":
            p = 3
        elif scheme_upper == "PCS":
            p = 4
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
            
        Wx = np.sinc(np.fft.fftfreq(N))
        Wy = np.sinc(np.fft.fftfreq(N))
        Wz = np.sinc(np.fft.rfftfreq(N))
        
        W_3d = Wx[:, None, None] * Wy[None, :, None] * Wz[None, None, :]
        W_3d = W_3d**p
        
        # Placeholder for aliasing correction (e.g., Jing 2005)
        # Future PRs will implement the sum over aliases. 
        # For now, aliasing_correction = 1.0.
        aliasing_correction = 1.0 
        
        # Protect against division by zero exactly at k=0 (though W(0)=1, just to be safe)
        W_sq = W_3d**2 * aliasing_correction
        valid_W = W_sq > 0
        P_k_3d[valid_W] /= W_sq[valid_W]

    # Binning in multiples of the fundamental frequency
    k_max = np.max(K_mag)
    if k_max == 0:
        max_multiple = 1
    else:
        max_multiple = int(np.ceil(k_max / kf))
        
    bins = np.arange(0.5, max_multiple + 1.5, 1.0) * kf
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    K_mag_flat = K_mag.flatten()
    P_k_flat = P_k_3d.flatten()
    
    # Determine weights for rfftn Hermitian symmetry
    weights = np.ones_like(K_mag_flat) * 2.0
    
    # k_z = 0 and k_z = Nyquist are not duplicated
    Nyq_z_idx = len(k_z) - 1 if N % 2 == 0 else -1
    z_idx = np.arange(len(k_z))
    is_single = (z_idx == 0)
    if Nyq_z_idx != -1:
        is_single = is_single | (z_idx == Nyq_z_idx)
        
    single_mask = is_single[None, None, :]
    single_mask_3d = np.broadcast_to(single_mask, K_mag.shape)
    weights[single_mask_3d.flatten()] = 1.0
    
    # Accumulate power and modes
    sum_P, _ = np.histogram(K_mag_flat, bins=bins, weights=P_k_flat * weights)
    count_modes, _ = np.histogram(K_mag_flat, bins=bins, weights=weights)
    
    valid = count_modes > 0
    Pk_monopole = np.zeros_like(sum_P)
    Pk_monopole[valid] = sum_P[valid] / count_modes[valid]
    
    k_3d = bin_centers[valid]
    pk_3d = Pk_monopole[valid]
    nmodes_3d = count_modes[valid].astype(int)
    
    return k_3d, pk_3d, nmodes_3d
