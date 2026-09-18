# `tud_lbm` developer documentation

If you're looking for user documentation, start with the
[Quickstart](../quickstart.rst) or the
[project README](https://github.com/szkuds/tud_lbm/blob/main/README.md).

## Development install

TUD-LBM requires Python 3.11 or newer.

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

Run the test suite with:

```shell
uv run pytest -v
```

Slow, full-pipeline tests are excluded from the default run. Include them with:

```shell
uv run pytest -m slow
```

The available markers (`unit`, `integration`, `conformance`, `slow`) are declared under
`[tool.pytest.ini_options]` in `pyproject.toml`.

### Test coverage

```shell
uv run pytest --cov --cov-report term --cov-report xml
```

The XML report must land at `coverage.xml` in the repository root — that is the path
`sonar.python.coverage.reportPaths` in `sonar-project.properties` points at, and it is what the
CI workflow produces. Use `--cov-report html` for a browsable report.

---

## Running linters locally

For linting and import sorting we use [ruff](https://docs.astral.sh/ruff/), and for type checking
[ty](https://github.com/astral-sh/ty). The full local quality gate is:

```shell
uv run ruff format
uv run ruff check          # add --fix to apply autofixes
uv run ty check
uv run pytest --cov --cov-report xml
```

`ruff` is configured with `select = ["ALL"]`; suppressions are narrowed per file under
`[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`. For `ty`, suppress with
`# ty: ignore[rule-code]` — **not** `# type: ignore` — and prefer fixing the underlying type error.

`ruff` and `ty` also run as [pre-commit](https://pre-commit.com/) hooks:

```shell
uv run pre-commit install
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
export UV_PROJECT_ENVIRONMENT=/scratch/<netid>/tud_lbm/.venv
```

Then reload:

```bash
source ~/.bashrc
```

### 5. Extract and sync

```bash
cd /scratch/<netid>
mkdir tud_lbm && tar -xzf tud_lbm.tar.gz -C tud_lbm
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

Login nodes have no GPU. DelftBlue GPU nodes use CUDA, and JAX's CPU and CUDA
builds are separate packages, so the variant is chosen at install time:

```bash
# CPU (compute-p1 and friends) — the default
scripts/setup_on_delftblue.sh

# CUDA (GPU nodes)
scripts/setup_on_delftblue.sh --cuda
```

`--cuda` runs `uv sync --extra cluster`, which pulls the `jax[cuda12]` wheels
declared in `pyproject.toml`. That extra carries a `sys_platform == 'linux'`
marker, so a plain `uv sync` on macOS resolves to CPU JAX and never sees the
CUDA packages.

Verify from a **GPU node**, not the login node — `jax.devices()` reports
`CpuDevice` on a login node no matter which build is installed:

```bash
srun --partition=<gpu-partition> --gpus-per-task=1 --time=00:05:00 \
     --account=<account> --pty \
     .venv/bin/python -c 'import jax; print(jax.devices())'
```

To submit a GPU job, set `SBATCH_GPUS_PER_TASK` in `scripts/db_defaults.env`
(or answer the `gpus-per-task` prompt in `scripts/db_new_job.sh`) together with
a GPU partition. The partition name and whether your account carries a GPU
entitlement are cluster facts, not repo facts — read them off `sinfo -o "%20P
%10G"` and `sacctmgr show assoc user=$USER format=account,partition,qos`.

---

## Operator registry & architecture

All operators (collision schemes, macroscopic solvers, forces, boundary conditions, lattice models, initialisers, …) are registered in a **single global registry** (`OPERATOR_REGISTRY` in `src/registry.py`) at import time via the `@register_operator` decorator. The registry supports both **pure functions** and **classes** as targets. Adding a new operator requires only the decorator — no factory, config, or CLI code changes.

See [Operators and the Registry](../operators.rst) for the full catalogue of registered operators and the per-kind decorators.

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

Supported operator kinds: `analysis` · `boundary_condition` · `collision_models` · `differential` · `eos` · `equilibrium` · `extra_state` · `force` · `initialise` · `lattice` · `macroscopic` · `obstacle` · `plotting` · `simulation_type` · `stream` · `update_timestep` · `wetting`

You can list all registered operators from the command line:

```shell
tud-lbm run --list-simulation-operators
tud-lbm run --list-simulation-analysis
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
from src.simulation_io.save import SimulationIO

io = SimulationIO(base_dir=config.results_dir,
                  config=config,
                  simulation_name=config.simulation_name,
                  output_format=config.output_format)
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

The codebase is organised into the following top-level packages under `src/`.
The per-operator catalogue is **not** duplicated here — it lives in
[Operators and the Registry](../operators.rst), which is generated from the
same names the registry resolves at runtime.

### `config` — Configuration

| Module              | Public API                                                    | Description                                                                                                                    |
| ------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `simulation_config` | `SimulationConfig`                                            | **Primary configuration dataclass.** Frozen, validated in `__post_init__`, serialisable. Never enters a JIT boundary.          |
| `adapter_base`      | `ConfigAdapter`, `get_adapter`                                | Abstract base for config adapters; `get_adapter()` dispatches on file extension.                                               |
| `adapter_toml`      | `TomlAdapter`                                                 | Reads and writes `.toml` configuration files.                                                                                  |
| `adapter_dict`      | `DictAdapter`                                                 | Reads configuration from plain Python dicts (flat or nested).                                                                  |
| `array_expansion`   | `expand_config`, `ArrayParameterSet`                          | Expands list-valued fields into a Cartesian product of `SimulationConfig` objects for parameter sweeps.                        |
| `config_overview`   | `ENABLE_X64`, `DISABLE_JIT`, `DEBUG_FLAG`, `BASE_RESULTS_DIR` | Canonical global flags and constants. `BASE_RESULTS_DIR` honours `TUD_LBM_DATA_DIR`.                                           |
| `jax_config`        | `configure_jax`                                               | Centralised JAX configuration (64-bit precision, JIT toggle).                                                                  |

#### `SimulationConfig` key fields

| Group                   | Fields                                                                                     | Defaults                                 |
| ----------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------- |
| **Identity**            | `sim_type`, `simulation_name`                                                              | `"single_phase"`, `None`                 |
| **Lattice & grid**      | `lattice_type`, `grid_shape`                                                               | `"D2Q9"`, `(64, 64)` → `(64, 64, 1)`     |
| **Time stepping**       | `nt`, `tau`                                                                                | `1000`, `1.0`                            |
| **Collision**           | `collision_scheme`, `k_diag`                                                               | `"bgk"`, `None`                          |
| **Boundary conditions** | `bc_config`                                                                                | periodic on all six faces                |
| **Obstacle**            | `obstacle_config`                                                                          | `None`                                   |
| **Wetting**             | `wetting_config`, `hysteresis_config`, `chemical_step_config`                              | `None`                                   |
| **Forces**              | `gravity_force`, `gravity_masked_force`, `electric_force`                                  | `None`                                   |
| **Initialisation**      | `init_type`, `init_dir`, `initialisation`                                                  | `"standard"`, `None`, `{}`               |
| **Output / IO**         | `results_dir`, `save_interval`, `skip_interval`, `save_fields`, `plot_fields`, `animate_fields`, `output_format` | `~/TUD_LBM_data`, `nt // 10`, `0`, `None`, `None`, `None`, `"numpy"` |
| **Multiphase**          | `eos`, `kappa`, `rho_l`, `rho_v`, `interface_width`, `g`, `a_eos`, `b_eos`, `r_eos`, `t_eos` | all `None`                             |
| **Extensible**          | `extra`                                                                                    | `{}`                                     |

Fields declared with `array_field()` are sweep-eligible; `wetting_config`,
`hysteresis_config`, `chemical_step_config`, and the force sections are also
sweepable one level deep. See [Adapters](../adapters.rst).

---

### `pipeline` — Simulation execution

| Module            | Public API                                     | Description                                                                                                                     |
| ----------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `setup`           | `SimulationSetup`, `build_setup`               | **Immutable `NamedTuple` operator container.** Built from `SimulationConfig`; holds pre-built operators, masks, and JIT scalars. |
| `runner`          | `run`, `init_state`                            | `lax.scan` time-stepping loop, in-memory or streaming-I/O mode.                                                                 |
| `parallel_runner` | `run_parallel_simulations`, `SimulationResult` | Runs multiple configs via `ProcessPoolExecutor`; returns per-run status, output dir, parameters, error, and duration.           |
| `state/`          | `State`, `WettingState`                        | `NamedTuple` pytrees carrying `f`, `rho`, `u`, `t`, forces, `h`, and wetting parameters.                                        |

---

### `operators` — LBM operators

Every operator registers itself at import time; each subpackage calls
`auto_load_operators()` so that dropping in a `_*.py` file is all that is
needed. `factory.build_operator(kind, name)` is the single lookup path, and
`protocols.py` holds the structural contracts.

Subpackages: `collision/` · `equilibrium/` · `macroscopic/` (with `eos/`) ·
`boundary/` · `streaming/` · `differential/` · `force/` · `initialise/` ·
`obstacle/` · `wetting/` (with `hysteresis/`) · `step/`.

The registered names for each kind, their parameters, and how to add one are
documented in [Operators and the Registry](../operators.rst).

---

### `lattice` — Lattice velocity models

`Lattice` (a `NamedTuple` pytree) and `build_lattice(name)` for `D2Q9` and
`D3Q19`. Array shapes and the 5-D layout convention are covered in
[Lattice and Array Conventions](../lattice.rst).

---

### `simulation_io` — I/O utilities

Note the package is `src.simulation_io`, **not** `src.io` — the latter
shadowed the stdlib `io` module once `src` became the import root.

| Module         | Description                                                                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `save`         | `SimulationIO` — creates the timestamped run directory, writes `config.toml`, `physical_parameters.txt`, and the log; binds the configured output writer.           |
| `callbacks`    | `jax.debug.callback` wrappers for streaming I/O inside `lax.scan`; used by `pipeline.runner.run()` when an `io_handler` is supplied.                                |
| `output_data/` | `output_writers` registry — `Numpy` (`.npz`) and `Vtk`. Subclassing `OutputWriter` registers a new format under its lower-cased class name.                         |
| `plotting/`    | `FigureBuilder`, `PlotOperator` (per-timestep panels), `AnalysisPlot` (snapshot-history figures), `Animator`, and `run_comparison` for cross-run plots.             |
| `readers/`     | Re-exports `DictAdapter` and `TomlAdapter`.                                                                                                                        |
| `analysis/`    | `droplet_metrics/` (the shared per-run metric layer), `stability`, `physical_parameters/`, `accelerations/`, `surface_tension/`, `wetting_debug`.                   |

Two invariants in the analysis layer are worth knowing before adding to it:
`compute_droplet_series` is the only code that reads `.npz` snapshots for
droplet metrics, and `build_simulation_csv` is the only code that decides
whether a run gets a `simulation_data.csv`. Details in
[Adapters](../adapters.rst).

**Plot-type rule:** any x–y or time-series data plot must be rendered with
`ax.scatter(...)`, never `ax.plot(...)`. Spatial field maps (`imshow`),
vector overlays (`quiver`), analytic fit lines drawn over scatter data, and
reference lines such as `axvline` are out of scope for this rule.

---

### `cli` — Command-line interface

The entry point is `tud-lbm = "src.cli.commands:cli"`. Importing
`src.cli.commands` registers the commands onto the group in `cli/app.py`;
importing `cli.app` alone yields an empty group.

| Command      | Description                                                                     |
| ------------ | --------------------------------------------------------------------------------- |
| `run`        | Run a simulation from `CONFIG_PATH`, or interactively when it is omitted.       |
| `visualise`  | Build static figures for a run directory. A group: `fields`, `analysis`, or both. |
| `animate`    | Encode saved snapshots to mp4/gif (needs the `animation` extra).                |
| `compare`    | Build CSV metrics and comparison plots for every run under a parent directory.  |
| `analyse`    | Standalone analyses for a config, e.g. `--surface-tension`.                     |
| `regime-map` | Classify the runs listed in a text file and plot Bo∥ against Oh.                |

Frequently used `run` options:

| Option                        | Description                                                             |
| ----------------------------- | ------------------------------------------------------------------------- |
| `--dry-run`                   | Parse and display the configuration without running.                    |
| `--override KEY=VALUE`        | Override a config key; values are parsed as TOML literals.              |
| `--no-prompt`                 | Skip interactive prompts.                                               |
| `--max-workers N`             | Parallel workers for a sweep.                                           |
| `--compare`                   | Build cross-run comparison plots after a sweep.                         |
| `--continue`                  | Resume from a previous run directory.                                   |
| `--init-dir` / `--init-wetting` | Seed initialisation from saved data / wetting initialisation.         |
| `--init-wetting-nt N`         | Length of the `--init-wetting` equilibration phase (default 50000).     |
| `--debug-stability`           | Enable NaN and checkerboard diagnostics.                                |
| `--debug-wetting`             | Enable wetting-optimiser diagnostics.                                   |
| `--debug-wetting-interval N`  | Timesteps between logged wetting rows (default 100; 1 logs every step).  |
| `--list-simulation-operators` | List registered physics operators and exit.                             |
| `--list-simulation-analysis`  | List registered analysis operators and exit.                            |

All commands share one error contract via the `cli_command` decorator in
`cli/_console.py`: `KeyboardInterrupt` exits 130, `click.UsageError` exits 2,
`SystemExit` passes through, and anything else prints a red `Error:` line and
exits 1 — or re-raises when `TUD_LBM_DEBUG` is set.

---

## Versioning

The project uses [calendar versioning](https://calver.org/) in the form `YYYY.MINOR.PATCH` — the
current version is `2026.0.1`.

Bumping the version across all files is done with
[bump-my-version](https://github.com/callowayproject/bump-my-version), e.g.

```shell
uv run bump-my-version bump minor  # bumps from e.g. 2026.0.1 to 2026.1.0
uv run bump-my-version bump patch  # bumps from e.g. 2026.0.1 to 2026.0.2
```

The version string is duplicated in four files — `src/__init__.py`, `pyproject.toml`,
`CITATION.cff` and `docs/conf.py` — listed under `[[tool.bumpversion.files]]` in `pyproject.toml`.
**Always bump with the tool rather than editing by hand.** Editing one file leaves the others
reporting a stale version, and the next `bump-my-version` run then fails to find the expected
string in them.

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
cd $(mktemp -d tud_lbm.XXXXXX)
git clone git@github.com:szkuds/tud_lbm .

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
