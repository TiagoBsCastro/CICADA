import numpy as np


def _validate_cubic_delta(delta, name):
    """Return a density field as an array after validating its shape."""
    delta = np.asarray(delta)
    if delta.ndim != 3:
        raise ValueError(f"{name} must be a 3D array")

    N = delta.shape[0]
    if delta.shape != (N, N, N):
        raise ValueError(f"{name} must be a cubic grid")

    return delta, N


def _interlaced_rfftn(delta_pair):
    """
    Return the interlaced Fourier-space density field.

    The second input field is assumed to be constructed from particle
    positions shifted by +0.5 cell in each Cartesian direction on the same
    fixed grid. With the NumPy FFT convention, this shifted field carries a
    phase exp(-i k.s). Multiplication by exp(+i k.s) aligns it with the
    unshifted field before averaging.
    """
    if not isinstance(delta_pair, (list, tuple)) or len(delta_pair) != 2:
        raise ValueError(
            "When interlacing=True, delta must be a tuple of two 3D arrays: "
            "(delta_original, delta_shifted)."
        )

    delta_1, N = _validate_cubic_delta(delta_pair[0], "delta[0]")
    delta_2, N_shifted = _validate_cubic_delta(delta_pair[1], "delta[1]")
    if N_shifted != N:
        raise ValueError("delta[0] and delta[1] must have the same shape")

    delta_k_1 = np.fft.rfftn(delta_1)
    delta_k_2 = np.fft.rfftn(delta_2)

    mx = np.fft.fftfreq(N) * N
    my = np.fft.fftfreq(N) * N
    mz = np.fft.rfftfreq(N) * N

    Mx, My, Mz = np.meshgrid(mx, my, mz, indexing="ij")
    phase = np.exp(1j * np.pi * (Mx + My + Mz) / N)

    return (delta_k_1 + delta_k_2 * phase) / 2.0, N


def _assignment_order(scheme):
    """Return the order of the mass-assignment window."""
    scheme_upper = str(scheme).upper()
    if scheme_upper == "NGP":
        return 1
    if scheme_upper == "CIC":
        return 2
    if scheme_upper == "TSC":
        return 3
    if scheme_upper == "PCS":
        return 4
    raise ValueError(f"Unknown scheme: {scheme}")


def _resolve_kmax(kmax, N, kf, K_mag):
    """Resolve a kmax specification into a physical wavenumber."""
    scalar_nyquist = (N // 2) * kf

    if kmax is None:
        value = np.max(K_mag)
    elif isinstance(kmax, str):
        spec = kmax.strip().lower().replace(" ", "")
        if spec in {"nyquist", "scalar_nyquist", "scalar-nyquist"}:
            value = scalar_nyquist
        elif spec in {"grid", "max", "all", "sqrt3nyquist", "sqrt(3)*nyquist"}:
            value = np.max(K_mag)
        elif spec.endswith("*nyquist"):
            factor = float(spec[: -len("*nyquist")])
            value = factor * scalar_nyquist
        else:
            value = float(spec)
    else:
        value = float(kmax)

    if not np.isfinite(value) or value <= 0:
        raise ValueError("kmax must resolve to a positive finite value")

    return float(value)


def power_spectrum_3d(delta, boxsize, scheme="CIC", interlacing=False, kmax="nyquist"):
    """
    Compute the auto-power spectrum of a 3D density field.

    Parameters
    ----------
    delta : array_like or tuple of array_like
        3D density contrast field. When ``interlacing=True``, this must be
        a tuple ``(delta, delta_shifted)`` where ``delta_shifted`` is the
        density field constructed from particle positions shifted by half
        a cell in each dimension, i.e. ``pos + 0.5 * boxsize / grid`` on
        the same fixed grid as ``delta``.
    boxsize : float
        Size of the simulation box.
    scheme : str, optional
        Mass-assignment scheme used to generate the density field
        ('NGP', 'CIC', 'TSC', 'PCS', 'None'). Default is 'CIC'.
    interlacing : bool, optional
        If True, combine the original and half-cell shifted density fields
        in Fourier space before computing the power spectrum. Default is
        False.
    kmax : {'nyquist', 'grid'} or float or str, optional
        Maximum Fourier-vector magnitude to include in the binning. The
        default, ``'nyquist'``, keeps only modes with ``|k|`` up to the
        scalar grid Nyquist. Use ``'grid'`` to recover the previous behavior,
        which bins modes up to the largest 3D grid-vector norm,
        approximately ``sqrt(3) * k_nyquist``. Strings of the form
        ``'0.7*nyquist'`` are also accepted.

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
        delta_k, N = _interlaced_rfftn(delta)
    else:
        delta, N = _validate_cubic_delta(delta, "delta")
        delta_k = np.fft.rfftn(delta)

    boxsize = float(boxsize)
    kf = 2.0 * np.pi / boxsize

    # Calculate frequencies.
    k_x = np.fft.fftfreq(N) * N * kf
    k_y = np.fft.fftfreq(N) * N * kf
    k_z = np.fft.rfftfreq(N) * N * kf

    Kx, Ky, Kz = np.meshgrid(k_x, k_y, k_z, indexing="ij")
    K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)

    # NumPy's forward FFT does not divide by N^3, so divide |delta_k|^2 by
    # N^6. Multiplying by boxsize^3 gives the usual dimensions of P(k).
    P_k_3d = (boxsize**3 / (N**6)) * np.abs(delta_k) ** 2

    # MAS window correction. This is the direct W^{-2}(k) correction, not an
    # explicit alias-sum or shot-noise correction.
    if scheme is not None and str(scheme).upper() != "NONE":
        p = _assignment_order(scheme)

        Wx = np.sinc(np.fft.fftfreq(N))
        Wy = np.sinc(np.fft.fftfreq(N))
        Wz = np.sinc(np.fft.rfftfreq(N))

        W_3d = Wx[:, None, None] * Wy[None, :, None] * Wz[None, None, :]
        W_3d = W_3d**p

        W_sq = W_3d**2
        valid_W = W_sq > 0
        P_k_3d[valid_W] /= W_sq[valid_W]

    # Binning in multiples of the fundamental frequency.
    kmax_value = _resolve_kmax(kmax, N, kf, K_mag)
    max_multiple = max(1, int(np.ceil(kmax_value / kf)))

    bins = np.arange(0.5, max_multiple + 1.5, 1.0) * kf
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    K_mag_flat = K_mag.ravel()
    P_k_flat = P_k_3d.ravel()

    # Determine weights for rfftn Hermitian symmetry.
    weights = np.ones_like(K_mag_flat) * 2.0

    # k_z = 0 and, for even N, k_z = Nyquist are not duplicated.
    Nyq_z_idx = len(k_z) - 1 if N % 2 == 0 else -1
    z_idx = np.arange(len(k_z))
    is_single = z_idx == 0
    if Nyq_z_idx != -1:
        is_single = is_single | (z_idx == Nyq_z_idx)

    single_mask = is_single[None, None, :]
    single_mask_3d = np.broadcast_to(single_mask, K_mag.shape)
    weights[single_mask_3d.ravel()] = 1.0

    include = K_mag_flat <= kmax_value
    K_mag_flat = K_mag_flat[include]
    P_k_flat = P_k_flat[include]
    weights = weights[include]

    # Accumulate power and modes.
    sum_P, _ = np.histogram(K_mag_flat, bins=bins, weights=P_k_flat * weights)
    count_modes, _ = np.histogram(K_mag_flat, bins=bins, weights=weights)

    valid = count_modes > 0
    Pk_monopole = np.zeros_like(sum_P)
    Pk_monopole[valid] = sum_P[valid] / count_modes[valid]

    k_3d = bin_centers[valid]
    pk_3d = Pk_monopole[valid]
    nmodes_3d = count_modes[valid].astype(int)

    return k_3d, pk_3d, nmodes_3d
