# AGENTS.md

This file defines the working instructions for coding agents contributing to CICADA.

## Project identity

CICADA stands for:

**Cosmological Infrastructure for Counts, Assignment, Density maps, and Analysis**

CICADA is intended to become a lightweight Python library for cosmological map-making and large-scale-structure statistics.

The project should be easier to install and maintain than performance-focused alternatives such as Pylians. The priority is correctness, clarity, testability, and standard dependencies.

## Current repository status

This repository is currently a planning scaffold.

Do not implement scientific routines unless explicitly instructed.

At this stage, agents may edit documentation, organize the repository, refine the roadmap, and prepare issue templates or planning documents.

## Core design goals

1. Use standard scientific Python dependencies.
2. Keep the public API simple.
3. Prefer readable code over premature optimization.
4. Validate all scientific routines with tests.
5. Make algorithms transparent enough for research use and teaching.
6. Avoid fragile compiled dependencies in the initial version.

## Allowed initial dependencies

The first implementation phase should use only:

- `numpy`
- `scipy`
- `matplotlib`
- `pytest`

Do not add any of the following unless explicitly requested:

- Cython
- Numba
- MPI
- OpenMP bindings
- Pylians
- h5py
- astropy
- pandas
- healpy
- jax
- cupy
- pytorch

## Planned first implementation phase

When implementation is requested, start with particle-to-grid and particle-to-map routines.

The planned public API is:

```python
density_grid_3d(
    pos,
    boxsize,
    grid,
    weights=None,
    scheme="CIC",
    periodic=True,
    overdensity=False,
    dtype=np.float32,
)
```

and:

```python
density_map_2d(
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
)
```

Do not add these functions until implementation is explicitly requested.

## Expected behavior for future density routines

When implemented:

- `pos` should be validated carefully.
- 3D routines should expect positions with shape `(N, 3)`.
- 2D routines should accept either `(N, 2)` or `(N, 3)`.
- `boxsize` may be a scalar or a sequence.
- `grid` may be an integer or a tuple.
- `weights` may be `None` or an array with shape `(N,)`.
- If `weights is None`, each particle contributes unit weight.
- If `periodic=True`, positions should be wrapped using `np.mod`.
- If `periodic=False`, particles outside the domain should be ignored.
- If `overdensity=True`, return `delta = rho / mean(rho) - 1`.
- Mass or count conservation must hold to numerical precision.

## Planned mass-assignment schemes

The first schemes should be:

1. NGP
2. CIC

For CIC:

- 3D particles contribute to 8 neighboring cells.
- 2D particles contribute to 4 neighboring cells.
- The assigned weights from each particle must sum to the particle weight.
- Periodic wrapping at boundaries must be handled explicitly.
- Prefer vectorized NumPy logic and `np.add.at` for the first clear version.

## Testing expectations

When implementation begins, add tests before or alongside the code.

Required tests:

1. Mass conservation in 2D NGP.
2. Mass conservation in 2D CIC.
3. Mass conservation in 3D NGP.
4. Mass conservation in 3D CIC.
5. Periodic wrapping.
6. Non-periodic particle rejection.
7. Weighted assignment preserves total weight.
8. 2D projection of 3D positions works as expected.
9. Slab selection works as expected.
10. Overdensity maps have mean close to zero when valid.

## Future modules

The library should be designed so that later modules can add:

- Fourier-space power spectra;
- cross-power spectra;
- real-space two-point correlation functions;
- projected correlation functions;
- smoothing and filtering;
- rebinning;
- shot-noise estimates;
- simple validation utilities.

## Coding style

Use formal, explicit, maintainable Python.

Every public function should eventually have:

- type hints where practical;
- a clear docstring;
- parameter descriptions;
- return-value description;
- shape conventions;
- informative `ValueError` messages for invalid input.

Avoid silent guessing. Validate assumptions.

## Performance policy

Correctness comes first.

Only optimize after:

1. tests exist;
2. the reference implementation is clear;
3. the bottleneck is measured;
4. the optimization does not make installation fragile.

## Documentation policy

Documentation should be concise but scientifically precise.

Every implemented feature should eventually have:

- a short explanation;
- a minimal example;
- at least one test;
- known limitations.
