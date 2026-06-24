import numpy as np

def _validate_pos(pos, ndim):
    pos = np.asarray(pos)
    if pos.ndim != 2 or pos.shape[1] < ndim:
        raise ValueError(f"pos must have shape (N, {ndim}) or greater")
    return pos

def _get_boxsize(boxsize, ndim):
    if np.isscalar(boxsize):
        return np.full(ndim, boxsize, dtype=np.float64)
    boxsize = np.asarray(boxsize, dtype=np.float64)
    if boxsize.shape != (ndim,):
        raise ValueError(f"boxsize must be a scalar or a sequence of length {ndim}")
    return boxsize

def _get_grid(grid, ndim):
    if np.isscalar(grid):
        return np.full(ndim, grid, dtype=int)
    grid = np.asarray(grid, dtype=int)
    if grid.shape != (ndim,):
        raise ValueError(f"grid must be a scalar or a sequence of length {ndim}")
    return grid

def density_grid_3d(
    pos,
    boxsize,
    grid,
    weights=None,
    scheme="CIC",
    periodic=True,
    overdensity=False,
    dtype=np.float32,
):
    """
    Construct a 3D density grid from particle positions.

    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Particle positions.
    boxsize : float or array_like of shape (3,)
        Size of the simulation box.
    grid : int or array_like of shape (3,)
        Number of grid cells per dimension.
    weights : array_like, shape (N,), optional
        Weights to assign to each particle.
    scheme : str, optional
        Mass assignment scheme ('NGP' or 'CIC'). Default is 'CIC'.
    periodic : bool, optional
        Whether to use periodic boundary conditions. Default is True.
    overdensity : bool, optional
        If True, return the overdensity delta = rho / mean(rho) - 1.
    dtype : data-type, optional
        Data type of the output grid.

    Returns
    -------
    grid_map : ndarray, shape (grid, grid, grid)
        The 3D density or overdensity grid.
    """
    pos = _validate_pos(pos, 3)[:, :3]
    boxsize = _get_boxsize(boxsize, 3)
    grid = _get_grid(grid, 3)
    
    N = pos.shape[0]
    if weights is None:
        weights = np.ones(N, dtype=dtype)
    else:
        weights = np.asarray(weights, dtype=dtype)
        if weights.shape != (N,):
            raise ValueError("weights must have shape (N,)")
            
    if periodic:
        pos = np.mod(pos, boxsize)
    else:
        valid = np.all((pos >= 0) & (pos < boxsize), axis=1)
        pos = pos[valid]
        weights = weights[valid]

    cell_size = boxsize / grid
    grid_map = np.zeros(tuple(grid), dtype=dtype)
    
    if len(pos) == 0:
        if overdensity:
            grid_map.fill(-1.0)
        return grid_map

    if scheme.upper() == "NGP":
        coords = np.floor(pos / cell_size).astype(int)
        if periodic:
            coords = np.mod(coords, grid)
        np.add.at(grid_map, tuple(coords.T), weights)
        
    elif scheme.upper() == "CIC":
        pos_grid = pos / cell_size - 0.5
        coords = np.floor(pos_grid).astype(int)
        d = pos_grid - coords
        t = 1.0 - d
        
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    weight_factor = (
                        (d[:, 0] if dx else t[:, 0]) *
                        (d[:, 1] if dy else t[:, 1]) *
                        (d[:, 2] if dz else t[:, 2])
                    )
                    
                    c_x = coords[:, 0] + dx
                    c_y = coords[:, 1] + dy
                    c_z = coords[:, 2] + dz
                    
                    if periodic:
                        c_x = np.mod(c_x, grid[0])
                        c_y = np.mod(c_y, grid[1])
                        c_z = np.mod(c_z, grid[2])
                    else:
                        valid = (
                            (c_x >= 0) & (c_x < grid[0]) &
                            (c_y >= 0) & (c_y < grid[1]) &
                            (c_z >= 0) & (c_z < grid[2])
                        )
                        c_xv = c_x[valid]
                        c_yv = c_y[valid]
                        c_zv = c_z[valid]
                        weight_factor_v = weight_factor[valid]
                        weights_v = weights[valid]
                        
                        np.add.at(grid_map, (c_xv, c_yv, c_zv), weights_v * weight_factor_v)
                        continue
                        
                    np.add.at(grid_map, (c_x, c_y, c_z), weights * weight_factor)
    elif scheme.upper() == "TSC":
        pos_grid = pos / cell_size - 0.5
        coords = np.floor(pos_grid + 0.5).astype(int)
        d = pos_grid - coords
        
        W_x = [0.5 * (0.5 - d[:, 0])**2, 0.75 - d[:, 0]**2, 0.5 * (0.5 + d[:, 0])**2]
        W_y = [0.5 * (0.5 - d[:, 1])**2, 0.75 - d[:, 1]**2, 0.5 * (0.5 + d[:, 1])**2]
        W_z = [0.5 * (0.5 - d[:, 2])**2, 0.75 - d[:, 2]**2, 0.5 * (0.5 + d[:, 2])**2]
        
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    weight_factor = W_x[dx+1] * W_y[dy+1] * W_z[dz+1]
                    
                    c_x = coords[:, 0] + dx
                    c_y = coords[:, 1] + dy
                    c_z = coords[:, 2] + dz
                    
                    if periodic:
                        c_x = np.mod(c_x, grid[0])
                        c_y = np.mod(c_y, grid[1])
                        c_z = np.mod(c_z, grid[2])
                    else:
                        valid = (
                            (c_x >= 0) & (c_x < grid[0]) &
                            (c_y >= 0) & (c_y < grid[1]) &
                            (c_z >= 0) & (c_z < grid[2])
                        )
                        c_xv = c_x[valid]
                        c_yv = c_y[valid]
                        c_zv = c_z[valid]
                        weight_factor_v = weight_factor[valid]
                        weights_v = weights[valid]
                        
                        np.add.at(grid_map, (c_xv, c_yv, c_zv), weights_v * weight_factor_v)
                        continue
                        
                    np.add.at(grid_map, (c_x, c_y, c_z), weights * weight_factor)

    elif scheme.upper() == "PCS":
        pos_grid = pos / cell_size - 0.5
        coords = np.floor(pos_grid).astype(int)
        d = pos_grid - coords
        
        W_x = [
            (1.0 - d[:, 0])**3 / 6.0,
            2.0/3.0 - d[:, 0]**2 + 0.5 * d[:, 0]**3,
            2.0/3.0 - (1.0 - d[:, 0])**2 + 0.5 * (1.0 - d[:, 0])**3,
            d[:, 0]**3 / 6.0
        ]
        W_y = [
            (1.0 - d[:, 1])**3 / 6.0,
            2.0/3.0 - d[:, 1]**2 + 0.5 * d[:, 1]**3,
            2.0/3.0 - (1.0 - d[:, 1])**2 + 0.5 * (1.0 - d[:, 1])**3,
            d[:, 1]**3 / 6.0
        ]
        W_z = [
            (1.0 - d[:, 2])**3 / 6.0,
            2.0/3.0 - d[:, 2]**2 + 0.5 * d[:, 2]**3,
            2.0/3.0 - (1.0 - d[:, 2])**2 + 0.5 * (1.0 - d[:, 2])**3,
            d[:, 2]**3 / 6.0
        ]
        
        for dx in (-1, 0, 1, 2):
            for dy in (-1, 0, 1, 2):
                for dz in (-1, 0, 1, 2):
                    weight_factor = W_x[dx+1] * W_y[dy+1] * W_z[dz+1]
                    
                    c_x = coords[:, 0] + dx
                    c_y = coords[:, 1] + dy
                    c_z = coords[:, 2] + dz
                    
                    if periodic:
                        c_x = np.mod(c_x, grid[0])
                        c_y = np.mod(c_y, grid[1])
                        c_z = np.mod(c_z, grid[2])
                    else:
                        valid = (
                            (c_x >= 0) & (c_x < grid[0]) &
                            (c_y >= 0) & (c_y < grid[1]) &
                            (c_z >= 0) & (c_z < grid[2])
                        )
                        c_xv = c_x[valid]
                        c_yv = c_y[valid]
                        c_zv = c_z[valid]
                        weight_factor_v = weight_factor[valid]
                        weights_v = weights[valid]
                        
                        np.add.at(grid_map, (c_xv, c_yv, c_zv), weights_v * weight_factor_v)
                        continue
                        
                    np.add.at(grid_map, (c_x, c_y, c_z), weights * weight_factor)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
        
    if overdensity:
        mean_rho = np.mean(grid_map)
        if mean_rho > 0:
            grid_map = grid_map / mean_rho - 1.0
        else:
            grid_map.fill(0.0)
            
    return grid_map

def density_map_2d(
    pos,
    boxsize,
    grid,
    weights=None,
    scheme="CIC",
    periodic=True,
    projection_axis=2,
    slab=None,
    overdensity=False,
    dtype=np.float32,
):
    """
    Construct a 2D density map from particle positions.

    Parameters
    ----------
    pos : array_like, shape (N, >=2)
        Particle positions.
    boxsize : float or array_like of shape (2,)
        Size of the simulation box in the projection plane.
    grid : int or array_like of shape (2,)
        Number of grid cells per dimension in the projection plane.
    weights : array_like, shape (N,), optional
        Weights to assign to each particle.
    scheme : str, optional
        Mass assignment scheme ('NGP' or 'CIC'). Default is 'CIC'.
    periodic : bool, optional
        Whether to use periodic boundary conditions. Default is True.
    projection_axis : int, optional
        Axis to project along (0, 1, or 2). Default is 2 (z-axis).
    slab : tuple of floats, optional
        (min, max) limits along the projection axis to include particles.
    overdensity : bool, optional
        If True, return the overdensity delta = rho / mean(rho) - 1.
    dtype : data-type, optional
        Data type of the output map.

    Returns
    -------
    grid_map : ndarray, shape (grid, grid)
        The 2D density or overdensity map.
    """
    pos = np.asarray(pos)
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError("pos must have shape (N, >=2)")
        
    axes = [i for i in range(pos.shape[1]) if i != projection_axis]
    if len(axes) > 2:
        axes = axes[:2]
    
    pos_2d = pos[:, axes]
    
    if slab is not None and pos.shape[1] > projection_axis:
        z_min, z_max = slab
        z_pos = pos[:, projection_axis]
        valid = (z_pos >= z_min) & (z_pos < z_max)
        pos_2d = pos_2d[valid]
        if weights is not None:
            weights = np.asarray(weights)[valid]
            
    boxsize_2d = _get_boxsize(boxsize, 2)
    grid_2d = _get_grid(grid, 2)
    
    if weights is None:
        weights = np.ones(pos_2d.shape[0], dtype=dtype)
    else:
        weights = np.asarray(weights, dtype=dtype)
        if weights.shape != (pos_2d.shape[0],):
            raise ValueError("weights must have shape corresponding to selected particles")
            
    if periodic:
        pos_2d = np.mod(pos_2d, boxsize_2d)
    else:
        valid = np.all((pos_2d >= 0) & (pos_2d < boxsize_2d), axis=1)
        pos_2d = pos_2d[valid]
        weights = weights[valid]
        
    cell_size = boxsize_2d / grid_2d
    grid_map = np.zeros(tuple(grid_2d), dtype=dtype)
    
    if len(pos_2d) == 0:
        if overdensity:
            grid_map.fill(-1.0)
        return grid_map
        
    if scheme.upper() == "NGP":
        coords = np.floor(pos_2d / cell_size).astype(int)
        if periodic:
            coords = np.mod(coords, grid_2d)
        np.add.at(grid_map, tuple(coords.T), weights)
        
    elif scheme.upper() == "CIC":
        pos_grid = pos_2d / cell_size - 0.5
        coords = np.floor(pos_grid).astype(int)
        d = pos_grid - coords
        t = 1.0 - d
        
        for dx in (0, 1):
            for dy in (0, 1):
                weight_factor = (
                    (d[:, 0] if dx else t[:, 0]) *
                    (d[:, 1] if dy else t[:, 1])
                )
                
                c_x = coords[:, 0] + dx
                c_y = coords[:, 1] + dy
                
                if periodic:
                    c_x = np.mod(c_x, grid_2d[0])
                    c_y = np.mod(c_y, grid_2d[1])
                else:
                    valid = (
                        (c_x >= 0) & (c_x < grid_2d[0]) &
                        (c_y >= 0) & (c_y < grid_2d[1])
                    )
                    c_xv = c_x[valid]
                    c_yv = c_y[valid]
                    weight_factor_v = weight_factor[valid]
                    weights_v = weights[valid]
                    
                    np.add.at(grid_map, (c_xv, c_yv), weights_v * weight_factor_v)
                    continue
                    
                np.add.at(grid_map, (c_x, c_y), weights * weight_factor)
    elif scheme.upper() == "TSC":
        pos_grid = pos_2d / cell_size - 0.5
        coords = np.floor(pos_grid + 0.5).astype(int)
        d = pos_grid - coords
        
        W_x = [0.5 * (0.5 - d[:, 0])**2, 0.75 - d[:, 0]**2, 0.5 * (0.5 + d[:, 0])**2]
        W_y = [0.5 * (0.5 - d[:, 1])**2, 0.75 - d[:, 1]**2, 0.5 * (0.5 + d[:, 1])**2]
        
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                weight_factor = W_x[dx+1] * W_y[dy+1]
                
                c_x = coords[:, 0] + dx
                c_y = coords[:, 1] + dy
                
                if periodic:
                    c_x = np.mod(c_x, grid_2d[0])
                    c_y = np.mod(c_y, grid_2d[1])
                else:
                    valid = (
                        (c_x >= 0) & (c_x < grid_2d[0]) &
                        (c_y >= 0) & (c_y < grid_2d[1])
                    )
                    c_xv = c_x[valid]
                    c_yv = c_y[valid]
                    weight_factor_v = weight_factor[valid]
                    weights_v = weights[valid]
                    
                    np.add.at(grid_map, (c_xv, c_yv), weights_v * weight_factor_v)
                    continue
                    
                np.add.at(grid_map, (c_x, c_y), weights * weight_factor)

    elif scheme.upper() == "PCS":
        pos_grid = pos_2d / cell_size - 0.5
        coords = np.floor(pos_grid).astype(int)
        d = pos_grid - coords
        
        W_x = [
            (1.0 - d[:, 0])**3 / 6.0,
            2.0/3.0 - d[:, 0]**2 + 0.5 * d[:, 0]**3,
            2.0/3.0 - (1.0 - d[:, 0])**2 + 0.5 * (1.0 - d[:, 0])**3,
            d[:, 0]**3 / 6.0
        ]
        W_y = [
            (1.0 - d[:, 1])**3 / 6.0,
            2.0/3.0 - d[:, 1]**2 + 0.5 * d[:, 1]**3,
            2.0/3.0 - (1.0 - d[:, 1])**2 + 0.5 * (1.0 - d[:, 1])**3,
            d[:, 1]**3 / 6.0
        ]
        
        for dx in (-1, 0, 1, 2):
            for dy in (-1, 0, 1, 2):
                weight_factor = W_x[dx+1] * W_y[dy+1]
                
                c_x = coords[:, 0] + dx
                c_y = coords[:, 1] + dy
                
                if periodic:
                    c_x = np.mod(c_x, grid_2d[0])
                    c_y = np.mod(c_y, grid_2d[1])
                else:
                    valid = (
                        (c_x >= 0) & (c_x < grid_2d[0]) &
                        (c_y >= 0) & (c_y < grid_2d[1])
                    )
                    c_xv = c_x[valid]
                    c_yv = c_y[valid]
                    weight_factor_v = weight_factor[valid]
                    weights_v = weights[valid]
                    
                    np.add.at(grid_map, (c_xv, c_yv), weights_v * weight_factor_v)
                    continue
                    
                np.add.at(grid_map, (c_x, c_y), weights * weight_factor)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
        
    if overdensity:
        mean_rho = np.mean(grid_map)
        if mean_rho > 0:
            grid_map = grid_map / mean_rho - 1.0
        else:
            grid_map.fill(0.0)
            
    return grid_map

def density_grid_3d_interlaced(
    pos,
    boxsize,
    grid,
    weights=None,
    scheme="CIC",
    periodic=True,
    overdensity=False,
    dtype=np.float32,
):
    """
    Construct a pair of 3D density grids for interlaced power spectrum estimation.

    The first grid is built from the original particle positions.  The second
    grid is built from positions shifted by +0.5 cells in every dimension.
    This shift convention matches the phase correction used by
    ``power_spectrum_3d(..., interlacing=True)``.

    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Particle positions.
    boxsize : float or array_like of shape (3,)
        Size of the simulation box.
    grid : int or array_like of shape (3,)
        Number of grid cells per dimension.
    weights : array_like, shape (N,), optional
        Weights to assign to each particle.
    scheme : str, optional
        Mass assignment scheme ('NGP', 'CIC', 'TSC', or 'PCS'). Default is 'CIC'.
    periodic : bool, optional
        Whether to use periodic boundary conditions. Default is True.
    overdensity : bool, optional
        If True, return overdensity delta = rho / mean(rho) - 1.
    dtype : data-type, optional
        Data type of the output grids.

    Returns
    -------
    delta : ndarray, shape (grid, grid, grid)
        Density grid from the original positions.
    delta_shifted : ndarray, shape (grid, grid, grid)
        Density grid from positions shifted by half a cell in each dimension.
    """
    pos = np.asarray(pos)
    boxsize_arr = _get_boxsize(boxsize, 3)
    grid_arr = _get_grid(grid, 3)

    cell_size = boxsize_arr / grid_arr
    shift = 0.5 * cell_size

    delta = density_grid_3d(
        pos, boxsize, grid,
        weights=weights, scheme=scheme, periodic=periodic,
        overdensity=overdensity, dtype=dtype,
    )

    pos_shifted = pos[:, :3] + shift
    if periodic:
        pos_shifted = np.mod(pos_shifted, boxsize_arr)

    delta_shifted = density_grid_3d(
        pos_shifted, boxsize, grid,
        weights=weights, scheme=scheme, periodic=periodic,
        overdensity=overdensity, dtype=dtype,
    )

    return delta, delta_shifted
