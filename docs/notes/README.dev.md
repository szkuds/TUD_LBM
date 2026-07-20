# `tud_lbm` developer documentation

If you're looking for user documentation, go [here](../../README.md).

## Development install

TUD-LBM requires Python 3.10 or newer.

We recommend using **uv** for development because it provides a fast, reproducible, and cross-platform workflow for editable installs and dependency management. For most contributors, uv is the simplest way to create an isolated development environment and install the project with extras.

### Install uv

Install `uv` by following the official instructions for your platform:

- https://docs.astral.sh/uv/getting-started/installation/

### Linux and macOS

```shell
git clone git@github.com:szkuds/tud_lbm.git
cd tud_lbm
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .[dev]
```

### Windows

```shell
git clone git@github.com:szkuds/tud_lbm.git
cd tud_lbm
uv venv --python 3.14
.venv\Scripts\activate
uv pip install -e .[dev]
```

### Documentation dependencies only

```shell
uv pip install -e .[docs]
```

---

## Running the tests

There are two ways to run tests.

The first way requires an activated virtual environment with the development tools installed:

```shell
pytest -v
```

The second is to use `tox`, which can be installed separately and can build the necessary virtual environments itself by simply running:

```shell
tox
```

### Test coverage

In an activated virtual environment with the development tools installed, inside the package directory, run:

```shell
coverage run
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.

---

## Running linters locally

For linting and import sorting we use [ruff](https://beta.ruff.rs/docs/):

```shell
ruff check .
ruff check . --fix
```

---

## Generating the API docs

```shell
cd docs
make html
```

The documentation will be in `docs/_build/html`.

If you do not have `make`, use:

```shell
sphinx-build -b html docs docs/_build/html
```

---

## Running on DelftBlue (HPC)

Since `git clone` is unavailable on DelftBlue login nodes, the workflow is:
**local archive → scp transfer → uv sync on cluster.**

### Prerequisites

Ensure the following files are committed in your local repo before archiving:

- `pyproject.toml`
- `uv.lock`
- `.python-version`

### 1. Local: create a clean archive

From your repo root:

```bash
git archive --format=tar.gz --output=tud_lbm.tar.gz HEAD
```

`git archive HEAD` produces a clean snapshot of committed files only — no `.venv`, no `__pycache__`, no `.git` bloat.

### 2. Transfer to DelftBlue

```bash
scp tud_lbm.tar.gz <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/
```

> **Use `/scratch`, not `/home`.** Package installs generate many small files and `/home` quota is limited.

### 3. On DelftBlue: install uv (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or follow the post-install instructions printed by the installer
```

### 4. Redirect uv cache to scratch (once)

Add these to your `~/.bashrc` to avoid filling `/home`:

```bash
export UV_CACHE_DIR=/scratch/<netid>/.uv_cache
export UV_PROJECT_ENVIRONMENT=/scratch/<netid>/src/.venv
```

Then reload:

```bash
source ~/.bashrc
```

### 5. Extract and sync

```bash
cd /scratch/<netid>
mkdir src && tar -xzf tud_lbm.tar.gz -C tud_lbm
cd tud_lbm

uv sync
```

`uv sync` installs the correct Python version and all dependencies from `uv.lock` in one step.

### Subsequent updates

After making changes locally:

```bash
# Local
git archive --format=tar.gz --output=tud_lbm.tar.gz HEAD
scp tud_lbm.tar.gz <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/

# On DelftBlue
cd /scratch/<netid>/tud_lbm
tar -xzf ../tud_lbm.tar.gz   # overwrites changed files in place
uv sync                        # no-op if uv.lock hasn't changed
```

### JAX CPU/GPU note

Login nodes have no GPU. DelftBlue GPU nodes use CUDA. JAX's CPU and GPU builds are separate, so verify your `pyproject.toml` pins the correct variant for the cluster environment before syncing.

Consider using a dependency group or environment marker to separate local (CPU) and cluster (CUDA) JAX installs if you run on both.

---

## Operator registry & architecture

All operators (collision schemes, macroscopic solvers, forces, boundary conditions, lattice models, initialisers, …) are registered in a **single global registry** (`OPERATOR_REGISTRY` in `tud_lbm/registry.py`) at import time via the `@register_operator` decorator. The registry supports both **pure functions** and **classes** as targets. Adding a new operator requires only the decorator — no factory, config, or CLI code changes.

**Pure function example (preferred):**

```python
from src.registry import register_operator

@register_operator("collision_models")
def collide_bgk(f, feq, tau, source=None):
    ...
collide_bgk.name = "bgk"
```

**Class example:**

```python
from src.registry import register_operator

@register_operator("collision_models")
class MyCollision:
    name = "my_custom"

    def __init__(self, config): ...

    def __call__(self, f, feq, tau, source=None): ...
```

Supported operator kinds: `boundary_condition` · `collision_models` · `differential` · `equilibrium` · `force` · `initialise` · `macroscopic` · `simulation_type` · `stream` · `update_timestep` · `wetting`

You can list all registered operators from the command line:

```shell
tud-lbm run --list-operators
```

---

## Core workflow

The main simulation workflow is **Config → Setup → State → Run**:

```python
from src import SimulationConfig, build_setup, init_state, run

# 1. Create configuration
config = SimulationConfig(grid_shape=(64, 64), tau=0.8, nt=5000)

# 2. Build immutable operator container
setup = build_setup(config)

# 3. Initialise state
state = init_state(setup)

# 4. Run — returns (final_state, trajectory)
final_state, trajectory = run(setup, state, nt=config.nt)
```

For long production runs, use **streaming I/O** to avoid accumulating the full trajectory in device memory:

```python
from src.simulation_io import SimulationIO

io = SimulationIO(base_dir=config.results_dir,
                  config=config.to_dict(),
                  simulation_name=config.simulation_name)
final_state, _ = run(setup, state, nt=config.nt,
                     save_interval=config.save_interval,
                     io_handler=io)
```

For **parameter sweeps**, use the parallel runner:

```python
from src.config.adapter_toml import TomlAdapter
from src.config.array_expansion import expand_config
from src.pipeline.parallel_runner import run_parallel_simulations

adapter = TomlAdapter()
config_dict = adapter.load_raw("config_parallel.toml")
configs, metadata = expand_config(config_dict)

results = run_parallel_simulations(configs, max_workers=4, verbose=True)
```

---

## Package reference

The codebase is organised into the following top-level packages under `tud_lbm/`.

### `config` — Configuration

| Module              | Public API                                            | Description                                                                                                                                                     |
| ------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `simulation_config` | `SimulationConfig`                                    | **Primary configuration dataclass.** Frozen, validated, serialisable. Never enters a JIT boundary. Holds all physics, time-stepping, BC, I/O, and multiphase parameters. |
| `adapter_base`      | `ConfigAdapter`, `get_adapter`                        | Abstract base class for config file adapters. Dispatch by file extension with `get_adapter()`.                                                                  |
| `adapter_toml`      | `TomlAdapter`                                         | Reads and writes `.toml` configuration files.                                                                                                                   |
| `adapter_dict`      | `DictAdapter`                                         | Reads configuration from plain Python dicts.                                                                                                                    |
| `array_expansion`   | `expand_config`, `ArrayParameterSet`                  | Expands list-valued fields into a Cartesian product of `SimulationConfig` objects for parameter sweeps.                                                         |
| `config_overview`   | `ENABLE_X64`, `DISABLE_JIT`, `DEBUG_FLAG`, `BASE_RESULTS_DIR` | Canonical global configuration flags and constants.                                                                                                    |
| `jax_config`        | `configure_jax`                                       | Centralised JAX configuration (64-bit precision, JIT toggle).                                                                                                   |

#### `SimulationConfig` key fields

| Group                   | Fields                                                                    | Defaults                                    |
| ----------------------- | ------------------------------------------------------------------------- | ------------------------------------------- |
| **Identity**            | `sim_type`, `simulation_name`                                             | `"single_phase"`, `None`                    |
| **Lattice & Grid**      | `lattice_type`, `grid_shape`                                              | `"D2Q9"`, `(64, 64)`                        |
| **Time Stepping**       | `nt`, `tau`                                                               | `1000`, `1.0`                               |
| **Collision**           | `collision_scheme`, `k_diag`                                              | `"bgk"`, `None`                             |
| **Boundary Conditions** | `bc_config`                                                               | Periodic on all edges                       |
| **Force**               | `force_enabled`, `force_obj`                                              | `False`, `None`                             |
| **Initialisation**      | `init_type`, `init_dir`                                                   | `"standard"`, `None`                        |
| **Output / IO**         | `results_dir`, `save_interval`, `skip_interval`, `save_fields`            | `~/TUD_LBM_data/results`, `100`, `0`, `None`|
| **Multiphase**          | `eos`, `kappa`, `rho_l`, `rho_v`, `interface_width`, `bubble`, `g`, etc. | All `None`/`False`                          |
| **Extensible**          | `extra`                                                                   | `{}`                                        |

---

### `pipeline` — Simulation Execution

| Module            | Public API                                    | Description                                                                                                                                                          |
| ----------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `setup`           | `SimulationSetup`, `build_setup`              | **Immutable `NamedTuple` operator container** (JAX pytree). Built from `SimulationConfig` via `build_setup`. Holds operators, masks, and physics scalars for JIT.   |
| `runner`          | `run`, `init_state`                           | `lax.scan`-based time-stepping loop. Supports in-memory trajectory and streaming I/O modes. Call `init_state(setup)` first, then `run(setup, state, nt=...)`.       |
| `parallel_runner` | `run_parallel_simulations`, `SimulationResult`| Executes multiple `SimulationConfig` objects in parallel via `ProcessPoolExecutor`. Returns a list of `SimulationResult` objects with status and metadata.           |
| `state/`          | `State`, `WettingState`                       | Dataclasses representing simulation state (distribution function, density, velocity, wetting fields).                                                                |

---

### `operators` — LBM Operators

All operators register themselves at import time via `@register_operator(kind)`.

#### Collision (`collision_models`)

| Module   | Registry Name | Description                                                          |
| -------- | ------------- | -------------------------------------------------------------------- |
| `_bgk`   | `"bgk"`       | Bhatnagar–Gross–Krook single-relaxation-time collision operator.     |
| `_mrt`   | `"mrt"`       | Multiple-Relaxation-Time (MRT) collision operator for D2Q9.          |

#### Equilibrium (`equilibrium`)

| Registry Name | Description                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------------ |
| `"wb"`        | Weight-based equilibrium distribution function from density and velocity using lattice weights.  |

#### Streaming (`stream`)

| Registry Name | Description                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------- |
| `"standard"`  | Standard streaming operator. Propagates populations via array rolls.                              |

#### Macroscopic (`macroscopic`)

| Registry Name         | Description                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| `"standard"`          | Calculates macroscopic density and velocity from population distributions.                        |
| `"double-well"`       | Multiphase macroscopic operator using the double-well equation of state.                          |
| `"carnahan-starling"` | Multiphase macroscopic operator using the Carnahan–Starling equation of state.                    |

#### Boundary Conditions (`boundary_condition`)

| Registry Name   | Description                                                                                                    |
| --------------- | -------------------------------------------------------------------------------------------------------------- |
| `"bounce-back"` | Half-way bounce-back boundary condition.                                                                       |
| `"periodic"`    | No-op operator (periodicity is handled by streaming).                                                          |
| `"symmetry"`    | Mirror-symmetry boundary condition.                                                                            |

#### Force (`force`)

| Registry Name          | Description                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| `"composite"`          | Combines multiple force fields by superposition.                                              |
| `"gravity_multiphase"` | Constant gravitational force, supporting inclined domains via `inclination_angle_deg`.        |
| `"electric"`           | Electrical force with electric potential distribution.                                        |
| `"source_term"`        | Forcing source term for incorporating body forces into the collision step.                    |

#### Initialisation (`initialise`)

| Registry Name                          | Description                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------------- |
| `"standard"`                           | Uniform density and velocity; equilibrium distribution.                     |
| `"multiphase_bubble"`                  | Low-density bubble at domain centre (smooth `tanh` profile).                |
| `"multiphase_bubble_bot"`              | Low-density bubble near the bottom of the domain.                           |
| `"multiphase_bubbles"`                 | Two low-density bubbles side-by-side.                                       |
| `"multiphase_droplet_top"`             | High-density droplet near the top of the domain.                            |
| `"multiphase_droplet_variable_radius"` | High-density droplet with user-specified radius.                            |
| `"wetting"`                            | Droplet at the bottom wall for wetting simulations.                         |
| `"wetting_chemical_step"`              | Droplet at the bottom wall with a chemical step.                            |
| `"init_from_file"`                     | Loads `rho` and `u` from a saved `.npz` file and reconstructs equilibrium. |

#### Differential (`differential`)

| Registry Name | Description                                                                              |
| ------------- | ---------------------------------------------------------------------------------------- |
| `"gradient"`  | Spatial gradient of a scalar field via central finite differences.                       |
| `"laplacian"` | Laplacian using the 9-point isotropic stencil.                                           |

#### Wetting (`wetting`)

| Registry Name             | Description                                                              |
| ------------------------- | ------------------------------------------------------------------------ |
| `"contact_angle"`         | Calculates contact angles (left and right) from a density field.         |
| `"contact_line_location"` | Calculates contact line locations from density and angle data.           |

#### Step operators (`update_timestep`)

| Registry Name             | Description                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| `"single_phase"`          | Full single-phase LBM timestep: equilibrium → collision → streaming → BC → macroscopic.      |
| `"multiphase"`            | Full multiphase timestep; adds interparticle-force computation via the EOS macroscopic op.   |
| `"multiphase_hysteresis"` | Multiphase timestep with advancing/receding contact angle hysteresis.                         |
| `"multiphase_wetting"`    | Multiphase timestep variant for wetting simulations.                                          |

#### Operator protocols (`protocols.py`)

Structural `Protocol` types (`CollisionOperator`, `StreamingOperator`, `MacroscopicOperator`, `BoundaryOperator`, `EquilibriumOperator`, `StepOperator`, `InitialPopulationOperator`, `ExtraStatePlugin`, `HysteresisOperator`, `DifferentialOperator`) define the contract for each operator category. Use them for static type checking and loose coupling.

---

### `lattice` — Lattice velocity models

| Module    | Class     | Description                                                                                                   |
| --------- | --------- | ------------------------------------------------------------------------------------------------------------- |
| `lattice` | `Lattice`, `build_lattice` | Lattice velocity model. Constructs velocities (`c`), weights (`w`), opposite indices, and directional index sets for `D2Q9`, `D3Q19`, etc. |

---

### `io` — I/O utilities

| Module               | Description                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `save`               | `SimulationIO` — saves timestep data (`.npz`), configuration snapshots (`.toml`), and manages directory structure.  |
| `callbacks`          | `jax.debug.callback` wrappers for streaming I/O during `lax.scan`; called from `pipeline.runner.run()`.               |
| `output_data/`       | Output data formatting helpers.                                                                                       |
| `plotting/`          | Post-processing plotting utilities. Loads results and config from run directories.                                    |
| `readers/`           | Readers for loading saved simulation data.                                                                            |
| `analysis/`          | `stability` (NaN/checkerboard diagnostics), `physical_parameters/` (Bond/Ohnesorge overview), `accelerations/` (acceleration fitting + regime classification), `surface_tension/` (Young-Laplace calibration). |

---

### `cli` — Command-Line Interface

```console
tud-lbm [CONFIG_PATH] [OPTIONS]
```

| Option / Argument  | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `CONFIG_PATH`      | Optional path to a `.toml` configuration file. If omitted, launches interactive mode. |
| `--no-prompt`      | Skip interactive prompts and use defaults for missing values.                          |
| `--dry-run`        | Parse configuration and display summary without running the simulation.                |
| `--list-operators` | List all registered operators and exit.                                                |
| `--override`       | Override a config key at runtime, e.g. `--override tau=0.7`.                          |

---

## Versioning

Bumping the version across all files is done with [bump-my-version](https://github.com/callowayproject/bump-my-version), e.g.

```shell
bump-my-version bump major  # bumps from e.g. 0.3.2 to 1.0.0
bump-my-version bump minor  # bumps from e.g. 0.3.2 to 0.4.0
bump-my-version bump patch  # bumps from e.g. 0.3.2 to 0.3.3
```

---

## Making a release

### (1/3) Preparation

1. Verify that the information in [`CITATION.cff`](../../CITATION.cff) is correct.
2. Make sure the [version has been updated](#versioning).
3. Run the unit tests with `pytest -v`.

### (2/3) PyPI

In a new terminal:

```shell
# OPTIONAL: prepare a new directory with fresh git clone
cd $(mktemp -d src.XXXXXX)
git clone git@github.com:szkuds/src .

python -m pip install --upgrade pip
python -m pip install .[publishing]

# create the source distribution and the wheel
python -m build

# upload to test pypi instance (requires credentials)
python -m twine upload --repository testpypi dist/*
```

Visit [https://test.pypi.org/project/tud_lbm](https://test.pypi.org/project/tud_lbm) and verify the upload. Then publish to PyPI:

```shell
python -m twine upload dist/*
```

### (3/3) GitHub

Make a [release on GitHub](https://github.com/szkuds/tud_lbm/releases/new). The GitHub–Zenodo integration will create a DOI snapshot automatically.
