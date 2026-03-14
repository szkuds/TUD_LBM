# `tud_lbm` developer documentation

If you're looking for user documentation, go [here](README.md).

## Development install

### Recommended: Using Conda (preferred for scientific computing)

We recommend using **conda** for development, as it handles complex dependencies (JAX, scipy) efficiently:

```shell
# Clone the repository
git clone https://git@github.com:szkuds/tud_lbm.git
cd tud_lbm

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate tud-lbm

# Install the package in editable mode with development dependencies
pip install --no-cache-dir --editable .[dev]

# Or install documentation dependencies only
pip install --no-cache-dir --editable .[docs]
```

### Alternative: Using pip with a virtual environment

```shell
# Create a virtual environment
python -m venv env

# Activate virtual environment
source env/bin/activate

# Make sure to have a recent version of pip and setuptools
python -m pip install --upgrade pip setuptools

# (from the project root directory)
# Install tud_lbm as an editable package with development dependencies
python -m pip install --no-cache-dir --editable .[dev]

# Or install documentation dependencies only
python -m pip install --no-cache-dir --editable .[docs]
```

> **Note:** The pip path may take longer as complex packages like JAX are compiled from source. For scientific computing, conda is strongly recommended.

Afterwards check that the install directory is present in the `PATH` environment variable.

## Running the tests

There are two ways to run tests.

The first way requires an activated virtual environment with the development tools installed:

```shell
pytest -v
```

The second is to use `tox`, which can be installed separately (e.g. with `pip install tox`), i.e. not necessarily inside the virtual environment you use for installing `tud_lbm`, but then builds the necessary virtual environments itself by simply running:

```shell
tox
```

Testing with `tox` allows for keeping the testing environment separate from your development environment.
The development environment will typically accumulate (old) packages during development that interfere with testing; this problem is avoided by testing with `tox`.

### Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests.
In an activated virtual environment with the development tools installed, inside the package directory, run:

```shell
coverage run
```

This runs tests and stores the result in a `.coverage` file.
To see the results on the command line, run

```shell
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.## Running linters locally

For linting and sorting imports we will use [ruff](https://beta.ruff.rs/docs/). Running the linters requires an
activated virtual environment with the development tools installed.

```shell
# linter
ruff check .

# linter with automatic fixing
ruff check . --fix
```

To fix readability of your code style you can use [yapf](https://github.com/google/yapf).## Generating the API docs

```shell
cd docs
make html
```

The documentation will be in `docs/_build/html`

If you do not have `make` use

```shell
sphinx-build -b html docs docs/_build/html
```

To find undocumented Python objects run

```shell
cd docs
make coverage
cat _build/coverage/python.txt
```

To [test snippets](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) in documentation run

```shell
cd docs
make doctest
```

## Operator registry & architecture

All operators (collision schemes, macroscopic solvers, forces, boundary conditions, lattice models, initialisers, …) are registered in a **single global registry** (`OPERATOR_REGISTRY` in `src/registry.py`) at import time via decorators.  The registry supports both **pure functions** and **classes** as targets.  Adding a new operator requires only the decorator — no factory, config, or CLI code changes.

**Pure function example (preferred):**

```python
from registry import collision_model

@collision_model(name="mrt")
def collide_mrt(f, feq, tau, source=None, k_diag=None):
    ...
```

**Lattice model example:**

```python
from registry import lattice_operator

@lattice_operator(name="D2Q9", dim=2, q=9)
def _build_d2q9() -> Lattice:
    ...
```

Convenience decorators are available for each operator kind: `@collision_model`, `@boundary_condition`, `@macroscopic_operator`, `@equilibrium_operator`, `@stream_operator`, `@force_model`, `@initialise_operator`, `@wetting_operator`, `@lattice_operator`, `@simulation_type_operator`, `@update_timestep_operator`.

For the full developer guide — including how to add config keys, where to place files, and how validation works — see **[`dev_notes/OperatorRegistry.md`](dev_notes/OperatorRegistry.md)**.

You can list all registered operators from the command line:

```shell
tud-lbm --list-simulation_operators
```

## Versioning

Bumping the version across all files is done with [bump-my-version](https://github.com/callowayproject/bump-my-version), e.g.

```shell
bump-my-version bump major  # bumps from e.g. 0.3.2 to 1.0.0
bump-my-version bump minor  # bumps from e.g. 0.3.2 to 0.4.0
bump-my-version bump patch  # bumps from e.g. 0.3.2 to 0.3.3
```

## Making a release

This section describes how to make a release in 3 parts:

1. preparation
1. making a release on PyPI
1. making a release on GitHub

### (1/3) Preparation


1. Verify that the information in [`CITATION.cff`](CITATION.cff) is correct.
1. Make sure the [version has been updated](#versioning).
1. Run the unit tests with `pytest -v`

### (2/3) PyPI

In a new terminal:

```shell
# OPTIONAL: prepare a new directory with fresh git clone to ensure the release
# has the state of origin/main branch
cd $(mktemp -d tud_lbm.XXXXXX)
git clone git@github.com:szkuds/tud_lbm .

# make sure to have a recent version of pip and the publishing dependencies
python -m pip install --upgrade pip
python -m pip install .[publishing]

# create the source distribution and the wheel
python -m build

# upload to test pypi instance (requires credentials)
python -m twine upload --repository testpypi dist/*
```

Visit
[https://test.pypi.org/project/tud_lbm](https://test.pypi.org/project/tud_lbm)
and verify that your package was uploaded successfully. Keep the terminal open, we'll need it later.

In a new terminal, without an activated virtual environment or an env directory:

```shell
cd $(mktemp -d tud_lbm-test.XXXXXX)

# prepare a clean virtual environment and activate it
python -m venv env
source env/bin/activate

# make sure to have a recent version of pip and setuptools
python -m pip install --upgrade pip

# install from test pypi instance:
python -m pip -v install --no-cache-dir \
--index-url https://test.pypi.org/simple/ \
--extra-index-url https://pypi.org/simple tud_lbm
```

Check that the package works as it should when installed from pypitest.

Then upload to pypi.org with:

```shell
# Back to the first terminal,
# FINAL STEP: upload to PyPI (requires credentials)
python -m twine upload dist/*
```

### (3/3) GitHub

Don't forget to also make a [release on GitHub](https://github.com/szkuds/tud_lbm/releases/new).GitHub-Zenodo integration will also trigger Zenodo into making a snapshot of your repository and sticking a DOI on it.



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

| Class | Registry Name | Description                                                                                                                      |
|-------|---------------|----------------------------------------------------------------------------------------------------------------------------------|
| `Force` | — | Abstract base class for all forces. Holds a force array of shape `(nx, ny, 1, d)`.                                               |
| `CompositeForce` | `"composite"` | Combines multiple force fields by superposition. Allows gravitational, electrical, and other forces to work together.            |
| `GravityForceMultiphase` | `"gravity_multiphase"` | Constant gravitational force across the grid, supporting inclined domains via `inclination_angle_deg`.                           |
| `ElectricForce` | `"electric"` | Electrical force with electric potential distribution. Solves for the electric potential using a separate distribution function. |
| `SourceTerm` | `"source_term"` | Forcing source term for incorporating body forces into the collision step.                                                       |

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
