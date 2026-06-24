import numpy as np

def power_spectrum_3d(delta, boxsize, scheme="CIC", interlacing=False):
    """
    Compute the auto-power spectrum of a 3D density field.

    Parameters
    ----------
    delta : array_like or tuple of array_like
        3D density contrast field. When ``interlacing=True``, this must be
        a tuple ``(delta, delta_shifted)`` where ``delta_shifted`` is the
        density field constructed from particle positions shifted by half
        a cell in each dimension (i.e. ``pos + 0.5 * boxsize / grid``).
    boxsize : float
        Size of the simulation box.
    scheme : str, optional
        Mass-assignment scheme used to generate the density field
        ('NGP', 'CIC', 'TSC', 'PCS', 'None'). Default is 'CIC'.
    interlacing : bool, optional
        If True, apply interlacing correction using two density fields
        (original and half-cell shifted) to suppress aliasing from the
        mass-assignment scheme. Default is False.

    Returns
    -------
    k_3d : ndarray
        1D array of wavenumber bin centers.
    pk_3d : ndarray
        1D array of the monopole power spectrum P(k).
    nmodes_3d : ndarray
        1D array containing the number of modes in each k bin.
    """
    if interlacing:
        if not isinstance(delta, (list, tuple)) or len(delta) != 2:
            raise ValueError(
                "When interlacing=True, delta must be a tuple of two 3D "
                "arrays: (delta_original, delta_shifted)."
            )
        delta_1 = np.asarray(delta[0])
        delta_2 = np.asarray(delta[1])

        if delta_1.ndim != 3:
            raise ValueError("delta[0] must be a 3D array")
        if delta_2.ndim != 3:
            raise ValueError("delta[1] must be a 3D array")

        N = delta_1.shape[0]
        if delta_1.shape != (N, N, N):
            raise ValueError("delta[0] must be a cubic grid")
        if delta_2.shape != (N, N, N):
            raise ValueError("delta[1] must be a cubic grid")

        # FFT both fields
        delta_k_1 = np.fft.rfftn(delta_1)
        delta_k_2 = np.fft.rfftn(delta_2)

        # Phase correction to undo the half-cell shift in Fourier space.
        # A real-space shift of 0.5 cells along each axis maps to a phase
        # exp(-i pi m / N) per axis in the DFT, where m is the integer
        # frequency index.  To undo the shift we multiply by the conjugate
        # phase and then average with the unshifted transform.
        mx = np.fft.fftfreq(N) * N   # integer frequency indices
        my = np.fft.fftfreq(N) * N
        mz = np.fft.rfftfreq(N) * N

        Mx, My, Mz = np.meshgrid(mx, my, mz, indexing='ij')
        phase = np.exp(1j * np.pi * (Mx + My + Mz) / N)

        delta_k = (delta_k_1 + delta_k_2 * phase) / 2.0
    else:
        delta = np.asarray(delta)
        if delta.ndim != 3:
            raise ValueError("delta must be a 3D array")

        N = delta.shape[0]
        if delta.shape != (N, N, N):
            raise ValueError("delta must be a cubic grid")

        delta_k = np.fft.rfftn(delta)

    boxsize = float(boxsize)
    kf = 2.0 * np.pi / boxsize

    # Calculate frequencies
    k_x = np.fft.fftfreq(N) * N * kf
    k_y = np.fft.fftfreq(N) * N * kf
    k_z = np.fft.rfftfreq(N) * N * kf

    Kx, Ky, Kz = np.meshgrid(k_x, k_y, k_z, indexing='ij')
    K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)

    # Power spectrum amplitude
    # numpy forward FFT does not divide by N^3, so we divide the squared
    # amplitude by N^6.  We multiply by boxsize**3 (the volume) to get the
    # correct dimensions for P(k).
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

        W_sq = W_3d**2
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
