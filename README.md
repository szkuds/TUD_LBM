# TUD LBM

[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/szkuds/tud_lbm)
[![github license badge](https://img.shields.io/github/license/szkuds/tud_lbm)](https://github.com/szkuds/TUD_LBM?tab=Apache-2.0-1-ov-file)
[![RSD](https://img.shields.io/badge/rsd-tud_lbm-00a3e3.svg)](https://research-software-directory.org/software/tudlbm)
[![workflow pypi badge](https://img.shields.io/pypi/v/tud_lbm.svg?colorB=blue)](https://pypi.python.org/project/tud_lbm/)
[![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>)
[![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/11832/badge)](https://bestpractices.coreinfrastructure.org/projects/11832)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

[![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=szkuds_TUD_LBM&metric=alert_status)](https://sonarcloud.io/dashboard?id=szkuds_TUD_LBM)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=szkuds_TUD_LBM&metric=coverage)](https://sonarcloud.io/dashboard?id=szkuds_TUD_LBM)
[![Documentation Status](https://readthedocs.org/projects/tud_lbm/badge/?version=latest)](https://tud_lbm.readthedocs.io/en/latest/?badge=latest)
[![build](https://github.com/szkuds/tud_lbm/actions/workflows/build.yml/badge.svg)](https://github.com/szkuds/tud_lbm/actions/workflows/build.yml)
[![cffconvert](https://github.com/szkuds/tud_lbm/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/szkuds/tud_lbm/actions/workflows/cffconvert.yml)
[![sonarcloud](https://github.com/szkuds/tud_lbm/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/szkuds/tud_lbm/actions/workflows/sonarcloud.yml)

A lattice Boltzmann method package developed at Delft University of Technology. This package provides tools for simulating fluid dynamics using the lattice Boltzmann method, supporting both single-phase and multi-phase simulations.

The project setup is documented in [project_setup.md](project_setup.md).

## Installation

### Recommended: Using Conda (preferred for scientific computing)

We recommend using **conda** for installation, as it handles complex dependencies (JAX, scipy) efficiently:

```console
# Clone the repository
git clone git@github.com:szkuds/tud_lbm.git
cd tud_lbm

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate tud-lbm

# Install the package in editable mode
pip install -e .
```

### Alternative: Using pip

For a basic pip-based installation:

```console
git clone git@github.com:szkuds/tud_lbm.git
cd tud_lbm
python -m pip install .
```

Note: This may take longer as complex packages like JAX are compiled from source. For scientific computing, conda is strongly recommended.

### Why Conda?

- **Faster installation**: Pre-built binaries for JAX and scipy (no compilation)
- **Reproducibility**: Locked dependency versions in `environment.yml`
- **Reliability**: No dependency resolver hangs with complex packages
- **Research standard**: Used widely in the scientific computing community

## Documentation

For full documentation, visit [tud_lbm.readthedocs.io](https://tud_lbm.readthedocs.io/en/latest/).

## Quick Start

### Python API

```python
from app_setup import SimulationSetup
from runner import Run

setup = SimulationSetup(
    grid_shape=(100, 100),
    tau=0.6,
    nt=10000,
    save_interval=1000,
)
sim = Run(setup)
sim.run(verbose=True)
```

### TOML Configuration File

```console
tud_lbm example/config_simple.toml
```

### Interactive CLI

```console
tud_lbm  # launches interactive parameter prompts
```

---

## Package Reference

The codebase is organised into seven top-level packages. Each section below lists the modules, their public classes, and a summary drawn from the source docstrings.

### `app_setup` — Configuration & Registry

The configuration package provides the primary user-facing interface for setting up simulations.

| Module | Public API | Description |
|--------|-----------|-------------|
| `simulation_setup` | `SimulationSetup` | **Single public entry point** for configuring a simulation. A flat, validated dataclass that holds all physics, time-stepping, boundary-condition, initialisation, and I/O parameters. Multiphase fields are optional and only validated when `sim_type="multiphase"`. |
| `registry` | `register_operator`, `get_operators`, `OPERATOR_REGISTRY` | Global operator registry. Every operator registers itself via the `@register_operator` class decorator. Per-kind look-ups (e.g. "all collision operators") are derived dynamically — adding a new operator only requires defining the class with a `name` attribute and applying the decorator. |
| `adapter_base` | `ConfigAdapter`, `get_adapter` | Abstract base class for configuration file adapters. Each adapter reads a specific file format and returns a `SimulationSetup`. Use `get_adapter()` to obtain the right adapter for a given file path by extension. |
| `adapter_toml` | `TomlAdapter` | TOML configuration file adapter. Reads `.toml` files and returns a validated `SimulationSetup`. Supports `[simulation_type]`, `[multiphase]`, `[[force]]`, `[boundary_conditions]`, and `[output]` tables. |
| `dir_config` | `BASE_RESULTS_DIR` | Directory configuration constants. Default base directory for storing simulation results (`~/TUD_LBM_data/results`). |
| `jax_config` | `configure_jax`, `ENABLE_X64`, `DISABLE_JIT` | Centralised JAX configuration. Call `configure_jax()` at the start of your script to apply 64-bit precision and JIT settings. |
| `saving_config` | `DEFAULT_SAVE_FIELDS`, `AVAILABLE_FIELDS`, `FORCE_REGISTRY` | Saving configuration constants. Defines default fields to save (`rho`, `u`) and all available fields (`rho`, `u`, `force`, `force_ext`, `f`, `h`). |

#### `SimulationSetup` Fields

| Group | Fields | Defaults |
|-------|--------|----------|
| **Identity** | `sim_type`, `simulation_name` | `"single_phase"`, `None` |
| **Lattice & Grid** | `lattice_type`, `grid_shape` | `"D2Q9"`, `(64, 64)` |
| **Time Stepping** | `nt`, `tau` | `1000`, `1.0` |
| **Collision** | `collision_scheme`, `k_diag` | `"bgk"`, `None` |
| **Boundary Conditions** | `bc_config` | Periodic on all edges |
| **Force** | `force_enabled`, `force_obj` | `False`, `None` |
| **Initialisation** | `init_type`, `init_dir` | `"standard"`, `None` |
| **Output / IO** | `results_dir`, `save_interval`, `skip_interval`, `save_fields` | `~/TUD_LBM_data/results`, `100`, `0`, `None` |
| **Multiphase** | `eos`, `kappa`, `rho_l`, `rho_v`, `interface_width`, `bubble`, `rho_ref`, `g` | All `None`/`False` |
| **Extensible** | `extra` | `{}` |

---

### `runner` — Simulation Execution

Composes the factory, time-loop runner, and I/O handler into a single entry point.

| Module | Public API | Description |
|--------|-----------|-------------|
| `run` | `Run` | **Top-level entry point.** Accepts a `SimulationSetup`, creates the simulation via the factory, sets up I/O, and delegates to `SimulationRunner`. Call `Run(setup).run()` to execute. |
| `simulation_factory` | `SimulationFactory` | Factory that creates simulation instances from a `SimulationSetup`. Uses the global operator registry to resolve `sim_type` to the correct simulation class. |
| `simulation_runner` | `SimulationRunner` | Owns the time-stepping loop, NaN checking, and data saving. Delegates field initialisation and per-step updates to the simulation object, and persistence to the I/O handler. |
| `step_result` | `StepResult` | Standardised dataclass returned from each timestep. Contains: `f` (distribution function), `rho` (density), `u` (velocity), `force`, `force_ext`, and `h` (electric potential) — all optional except `f`. |

---

### `simulation_type` — Simulation Orchestrators

Each simulation type orchestrates the complete LBM workflow (operator setup, field initialisation, time-stepping).

| Module | Class | Registry Name | Description |
|--------|-------|---------------|-------------|
| `base` | `BaseSimulation` | — | Abstract base class. Provides shared helpers `_make_initialiser()` and `_make_boundary_condition()` that resolve operators from the registry. Defines the abstract interface: `setup_operators()`, `initialise_fields()`, `run_timestep()`. |
| `single_phase` | `SinglePhaseSimulation` | `"single_phase"` | Single-phase LBM simulation. Sets up collision, streaming, macroscopic, and boundary-condition operators. Supports optional external forces. |
| `multiphase` | `MultiphaseSimulation` | `"multiphase"` | Multiphase (two-phase) LBM simulation. Adds equation-of-state handling, surface tension (`kappa`), wetting detection, and selects between `UpdateMultiphase` and `UpdateMultiphaseHysteresis` based on config. |

---

### `simulation_operators` — LBM Operators

All operators register themselves via `@register_operator(kind)` and are resolved dynamically from the global registry. Each operator's `__call__` method is the primary interface.

#### Collision Models (`collision_models`)

| Class | Registry Name | Description |
|-------|---------------|-------------|
| `CollisionBase` | — | Abstract base class for collision operators. Provides shared lattice and grid setup. |
| `CollisionBGK` | `"bgk"` | Bhatnagar–Gross–Krook (BGK) single-relaxation-time collision operator. |
| `CollisionMRT` | `"mrt"` | Multiple-Relaxation-Time (MRT) collision operator. Uses a moment transformation matrix for D2Q9. |

#### Equilibrium (`equilibrium`)

| Class | Registry Name | Description |
|-------|---------------|-------------|
| `EquilibriumWB` | `"wb"` | Weight-based equilibrium distribution function. Computes the discrete equilibrium from density and velocity using lattice weights. |

#### Streaming (`stream`)

| Class | Registry Name | Description |
|-------|---------------|-------------|
| `Streaming` | `"standard"` | Standard streaming operator. Propagates populations along lattice velocity directions using array rolls. |

#### Macroscopic (`macroscopic`)

| Class | Registry Name | Description |
|-------|---------------|-------------|
| `Macroscopic` | `"standard"` | Calculates macroscopic density and velocity fields from the population distribution. |
| `MacroscopicMultiphaseDW` | `"double-well"` | Multiphase macroscopic operator using the double-well equation of state. Computes chemical potential, gradient, and Laplacian terms for the interparticle force. |
| `MacroscopicMultiphaseCS` | `"carnahan-starling"` | Multiphase macroscopic operator using the Carnahan–Starling equation of state. Inherits from `MacroscopicMultiphaseDW` and overrides the EOS. |

#### Boundary Conditions (`boundary_condition`)

| Class | Registry Name | Description |
|-------|---------------|-------------|
| `BoundaryConditionBase` | — | Abstract base class. Provides common grid, lattice, and validation for all BC operators. |
| `BoundaryCondition` | `"standard"` | **Composite dispatcher.** Inspects `bc_config` at construction time, resolves per-edge BC operators from the registry, and chains their `__call__` methods. No hardcoded type map needed. |
| `BounceBackBoundaryCondition` | `"bounce-back"` | Half-way bounce-back rule. Replaces each incoming distribution at the wall by its opposite-direction counterpart from the post-collision state. |
| `PeriodicBoundaryCondition` | `"periodic"` | No-op operator. Periodicity is handled by the streaming step; this exists so users can explicitly request `"periodic"` in configs. |
| `SymmetryBoundaryCondition` | `"symmetry"` | Mirror-symmetry rule. Replaces each incoming distribution at the wall by the mirrored distribution from the post-collision state. |
| `WettingBoundaryCondition` | `"wetting"` | Wetting boundary (bounce-back variant). Registered as `"wetting"` so that config files using `bottom = "wetting"` resolve via the registry. |

#### Force (`force`)

| Class | Registry Name | Description |
|-------|---------------|-------------|
| `Force` | — | Abstract base class for all forces. Holds a force array of shape `(nx, ny, 1, d)`. |
| `CompositeForce` | `"composite"` | Combines multiple force fields by superposition. Allows gravitational, electrical, and other forces to work together. |
| `GravityForceMultiphase` | `"gravity_multiphase"` | Constant gravitational force across the grid, supporting inclined domains via `inclination_angle_deg`. |
| `ElectricForce` | `"electric"` | Electrical force with electric potential distribution. Solves for the electric potential using a separate distribution function. |
| `SourceTerm` | `"source_term"` | Guo forcing source term for incorporating body forces into the collision step. |

#### Initialisation (`initialise`)

| Class | Registry Name | Description |
|-------|---------------|-------------|
| `InitialisationBase` | — | Abstract base class. Provides common grid, lattice, and equilibrium setup. Subclasses override `__call__` to produce the initial distribution function. |
| `StandardInitialisation` | `"standard"` | Uniform density and velocity field. Returns the equilibrium distribution for single-phase simulations. |
| `InitialiseMultiphaseBubble` | `"multiphase_bubble"` | Low-density bubble in the domain centre surrounded by liquid (smooth `tanh` profile). |
| `InitialiseMultiphaseBubbleBot` | `"multiphase_bubble_bot"` | Low-density bubble near the bottom of the domain surrounded by liquid. |
| `InitialiseMultiphaseBubbleBubble` | `"multiphase_bubble_bubble"` | Two low-density bubbles side-by-side surrounded by liquid. |
| `InitialiseMultiphaseDroplet` | `"multiphase_droplet"` | High-density droplet in the domain centre surrounded by vapour. |
| `InitialiseMultiphaseDropletTop` | `"multiphase_droplet_top"` | High-density droplet near the top of the domain surrounded by vapour. |
| `InitialiseMultiphaseDropletVariableRadius` | `"multiphase_droplet_variable_radius"` | High-density droplet with user-specified radius. |
| `InitialiseMultiphaseLateralBubble` | `"multiphase_lateral_bubble"` | Two low-density bubbles stacked vertically surrounded by liquid. |
| `InitialiseWetting` | `"wetting"` | Droplet at the bottom wall for wetting simulations (smooth `tanh` profile centred horizontally). |
| `InitialiseWettingChemicalStep` | `"wetting_chemical_step"` | Droplet at the bottom wall with a chemical step. The droplet is offset horizontally. |
| `InitialiseFromFile` | `"init_from_file"` | Loads `rho` and `u` from a saved `.npz` file and reconstructs the equilibrium distribution. |

#### Differential (`differential`)

| Class | Registry Name | Description |
|-------|---------------|-------------|
| `Gradient` | `"gradient"` | Computes the spatial gradient of a scalar field using central finite differences with configurable padding modes for boundary conditions. |
| `Laplacian` | `"laplacian"` | Computes the Laplacian of a scalar field using the 9-point isotropic stencil with configurable padding modes. |

#### Wetting (`wetting`)

| Class | Registry Name | Description |
|-------|---------------|-------------|
| `ContactAngle` | `"contact_angle"` | Calculates contact angles (left and right) from a density field. |
| `ContactLineLocation` | `"contact_line_location"` | Calculates contact line locations (left and right) from density and angle data. |

---

### `simulation_domain` — Grid & Lattice

| Module | Class | Description |
|--------|-------|-------------|
| `grid.grid` | `Grid` | Simulation domain grid. Stores shape and dimension, providing edge extraction utilities. Supports 2D `(nx, ny)` and 3D `(nx, ny, nz)` grids. |
| `lattice.lattice` | `Lattice` | Lattice velocity model. Constructs lattice velocities (`c`), weights (`w`), opposite indices, and directional index sets for a given lattice type (`D2Q9`, `D3Q19`). |

---

### `update_timestep` — Per-Step Update Operators

| Class | Registry Name | Description |
|-------|---------------|-------------|
| `Update` | `"single_phase"` | Full single-phase LBM timestep: equilibrium → collision → streaming → boundary conditions → macroscopic. |
| `UpdateMultiphase` | `"multiphase"` | Full multiphase LBM timestep. Extends `Update` with interparticle-force computation via the macroscopic EOS operator. |
| `UpdateMultiphaseHysteresis` | `"multiphase_hysteresis"` | Multiphase timestep with contact angle hysteresis. Extends `UpdateMultiphase` with advancing/receding contact angle enforcement and parameter optimisation. |

---

### `util` — I/O & Plotting Utilities

| Module | Description |
|--------|-------------|
| `io` | `SimulationIO` — handles saving timestep data (`.npz` files), configuration snapshots, and directory management. |
| `plotting` | Post-processing plotting utilities. Loads simulation results and configuration from run directories for visualisation. |

---

### `cli` — Command-Line Interface

```console
tud_lbm [CONFIG_PATH] [--no-prompt] [--dry-run] [--list-simulation_operators]
```

| Option | Description |
|--------|-------------|
| `CONFIG_PATH` | Optional path to a `.toml` configuration file. If omitted, launches interactive mode. |
| `--no-prompt` | Skip interactive prompts and use defaults for missing values. |
| `--dry-run` | Parse configuration and display summary without running the simulation. |
| `--list-simulation_operators` | List all registered operators and exit. |

---

### Operator Registry

All operators self-register via the `@register_operator(kind)` decorator. The supported operator kinds are:

`boundary_condition` · `collision_models` · `differential` · `equilibrium` · `force` · `initialise` · `macroscopic` · `simulation_type` · `stream` · `update_timestep` · `wetting`

To add a new operator, create a class with a `name` attribute and decorate it:

```python
from app_setup.registry import register_operator

@register_operator("collision_models")
class MyCustomCollision:
    name = "my_custom"

    def __init__(self, config):
        ...

    def __call__(self, f, feq, source):
        ...
```

The operator is then automatically available in `get_operators("collision_models")` and accepted by `SimulationSetup` validation.

## Contributing

If you want to contribute to the development of tud_lbm,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
