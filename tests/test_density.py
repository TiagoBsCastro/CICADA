import numpy as np
import pytest
from cicada.density import density_grid_3d, density_map_2d

def test_mass_conservation_3d_ngp():
    pos = np.random.uniform(0, 10, (100, 3))
    grid = density_grid_3d(pos, boxsize=10.0, grid=16, scheme="NGP", periodic=True)
    assert np.isclose(grid.sum(), 100.0)

def test_mass_conservation_3d_cic():
    pos = np.random.uniform(0, 10, (100, 3))
    grid = density_grid_3d(pos, boxsize=10.0, grid=16, scheme="CIC", periodic=True)
    assert np.isclose(grid.sum(), 100.0)

def test_mass_conservation_3d_tsc():
    pos = np.random.uniform(0, 10, (100, 3))
    grid = density_grid_3d(pos, boxsize=10.0, grid=16, scheme="TSC", periodic=True)
    assert np.isclose(grid.sum(), 100.0)

def test_mass_conservation_3d_pcs():
    pos = np.random.uniform(0, 10, (100, 3))
    grid = density_grid_3d(pos, boxsize=10.0, grid=16, scheme="PCS", periodic=True)
    assert np.isclose(grid.sum(), 100.0)

def test_mass_conservation_2d_ngp():
    pos = np.random.uniform(0, 10, (100, 2))
    grid = density_map_2d(pos, boxsize=10.0, grid=16, scheme="NGP", periodic=True)
    assert np.isclose(grid.sum(), 100.0)

def test_mass_conservation_2d_cic():
    pos = np.random.uniform(0, 10, (100, 2))
    grid = density_map_2d(pos, boxsize=10.0, grid=16, scheme="CIC", periodic=True)
    assert np.isclose(grid.sum(), 100.0)

def test_mass_conservation_2d_tsc():
    pos = np.random.uniform(0, 10, (100, 2))
    grid = density_map_2d(pos, boxsize=10.0, grid=16, scheme="TSC", periodic=True)
    assert np.isclose(grid.sum(), 100.0)

def test_mass_conservation_2d_pcs():
    pos = np.random.uniform(0, 10, (100, 2))
    grid = density_map_2d(pos, boxsize=10.0, grid=16, scheme="PCS", periodic=True)
    assert np.isclose(grid.sum(), 100.0)

def test_periodic_wrapping():
    # A single particle placed outside the box
    pos = np.array([[12.0, -2.0, 5.0]])
    grid_ngp = density_grid_3d(pos, boxsize=10.0, grid=10, scheme="NGP", periodic=True)
    assert np.isclose(grid_ngp.sum(), 1.0)
    
    # In NGP with grid 10, box 10 -> cell size 1
    # 12 -> 2, -2 -> 8, 5 -> 5
    assert grid_ngp[2, 8, 5] == 1.0

def test_non_periodic_rejection():
    # Particles outside the box should be ignored when periodic=False
    pos = np.array([[12.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    grid_ngp = density_grid_3d(pos, boxsize=10.0, grid=10, scheme="NGP", periodic=False)
    assert np.isclose(grid_ngp.sum(), 1.0) # Only the particle inside should be counted

def test_weighted_assignment():
    pos = np.array([[5.0, 5.0, 5.0], [2.0, 2.0, 2.0]])
    weights = np.array([3.0, 7.0])
    grid = density_grid_3d(pos, boxsize=10.0, grid=10, weights=weights, scheme="CIC", periodic=True)
    assert np.isclose(grid.sum(), 10.0)

def test_2d_projection():
    pos = np.array([[5.0, 4.0, 3.0]])
    # Project along axis 2 (z)
    grid = density_map_2d(pos, boxsize=10.0, grid=10, projection_axis=2, scheme="NGP")
    assert np.isclose(grid.sum(), 1.0)
    assert grid[5, 4] == 1.0
    
    # Project along axis 0 (x) -> y and z remain
    grid2 = density_map_2d(pos, boxsize=10.0, grid=10, projection_axis=0, scheme="NGP")
    assert grid2[4, 3] == 1.0

def test_slab_selection():
    pos = np.array([[5.0, 5.0, 2.0], [5.0, 5.0, 8.0]])
    # Only keep particles with z between 0 and 5
    grid = density_map_2d(pos, boxsize=10.0, grid=10, slab=(0.0, 5.0))
    assert np.isclose(grid.sum(), 1.0)

def test_overdensity():
    pos = np.random.uniform(0, 10, (100, 3))
    grid = density_grid_3d(pos, boxsize=10.0, grid=10, overdensity=True)
    assert np.isclose(grid.mean(), 0.0, atol=1e-6)
    
    # Test empty box
    empty_grid = density_grid_3d(np.empty((0, 3)), boxsize=10.0, grid=10, overdensity=True)
    assert np.all(empty_grid == -1.0)
