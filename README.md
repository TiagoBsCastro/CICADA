# CICADA

<p align="center">
  <img src="assets/cicada-logo.png" alt="CICADA logo" width="420">
</p>

**CICADA** stands for **Cosmological Infrastructure for Counts, Assignment, Density maps, and Analysis**.

CICADA is planned as a lightweight Python library for cosmological map-making and large-scale-structure statistics. The motivation is to provide a simpler, easier-to-install alternative to performance-focused libraries such as Pylians, while keeping the code readable, testable, and based on standard scientific Python tools.

## Motivation

Many cosmological analysis workflows need simple operations such as assigning particles to grids, creating projected density maps, and computing two-point statistics. Existing tools are often highly optimized, but may depend on compiled extensions or dependency stacks that are difficult to maintain across systems.

CICADA aims to start from a different point:

- simple installation;
- standard dependencies;
- clear APIs;
- careful tests;
- transparent algorithms;
- performance improvements only after correctness is established.

## Initial scientific scope

The first implementation phase will focus on density-field construction:

- 3D density grids from particle positions;
- 2D projected density maps;
- periodic and non-periodic boundary handling;
- unweighted particle counts;
- mass-weighted assignment;
- NGP and CIC assignment schemes;
- conversion from counts or mass density to overdensity contrast.

## Future scope

Future development may include:

- Fourier-space two-point statistics;
- real-space two-point correlation functions;
- auto- and cross-power spectra;
- smoothing and filtering;
- shot-noise estimates;
- simple plotting utilities;
- validation against reference implementations.

## Planned dependencies

The initial version should use only standard scientific Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `pytest`

Additional dependencies should only be introduced when there is a clear reason.

## Repository layout

```text
cicada-lss/
├── AGENTS.md
├── README.md
├── LICENSE
├── .gitignore
└── assets/
    └── cicada-logo.png
```

There is intentionally no `src/`, `cicada/`, or scientific implementation yet.

## Development principle

The first rule of CICADA is:

> Prefer transparent, correct, well-tested code before optimization.

This project should remain easy to inspect, easy to install, and easy to teach from.
