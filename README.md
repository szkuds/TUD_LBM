# TUD LBM

[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/szkuds/tud_lbm)
[![github license badge](https://img.shields.io/github/license/szkuds/tud_lbm)](https://github.com/szkuds/TUD_LBM?tab=Apache-2.0-1-ov-file)
[![RSD](https://img.shields.io/badge/rsd-tud_lbm-00a3e3.svg)](https://research-software-directory.org/software/tudlbm)
[![workflow pypi badge](https://img.shields.io/pypi/v/tud_lbm.svg?colorB=blue)](https://pypi.python.org/project/tud_lbm/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19204133.svg)](https://doi.org/10.5281/zenodo.19204133)
[![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/11832/badge)](https://bestpractices.coreinfrastructure.org/projects/11832)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

[![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=szkuds_TUD_LBM&metric=alert_status)](https://sonarcloud.io/dashboard?id=szkuds_TUD_LBM)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=szkuds_TUD_LBM&metric=coverage)](https://sonarcloud.io/dashboard?id=szkuds_TUD_LBM)
[![Documentation Status](https://readthedocs.org/projects/tud-lbm/badge/?version=latest)](https://tud-lbm.readthedocs.io/en/latest/?badge=latest)
[![build](https://github.com/szkuds/tud_lbm/actions/workflows/build.yml/badge.svg)](https://github.com/szkuds/tud_lbm/actions/workflows/build.yml)
[![cffconvert](https://github.com/szkuds/tud_lbm/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/szkuds/tud_lbm/actions/workflows/cffconvert.yml)
[![sonarcloud](https://github.com/szkuds/tud_lbm/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/szkuds/tud_lbm/actions/workflows/sonarcloud.yml)

A JAX-accelerated lattice Boltzmann method framework developed at Delft University of Technology. TUD-LBM supports single-phase, multiphase, wetting, hysteresis, and chemical-step simulations through a configuration-driven workflow.

## Installation

TUD-LBM requires Python 3.11 or newer. We recommend [uv](https://docs.astral.sh/uv/getting-started/installation/) for reproducible environments and dependency management.

```console
git clone https://github.com/szkuds/tud_lbm.git
cd tud_lbm
uv sync
uv run tud-lbm --help
```

For development dependencies, use `uv sync --extra dev`.

## Documentation

For full documentation, visit [tud-lbm.readthedocs.io](https://tud-lbm.readthedocs.io/en/latest/).

## Quick Start

### Python API

```python
from src import SimulationConfig, build_setup, init_state, run

config = SimulationConfig(grid_shape=(64, 64), tau=0.8, nt=1_000)
setup = build_setup(config)
state = init_state(setup)
final_state, trajectory = run(setup, state)
```

`SimulationConfig` supplies a D2Q9 lattice, BGK collision, periodic boundaries, and rest-equilibrium initialisation by default. `final_state` is the state after the requested time steps; `trajectory` contains sampled in-memory states.

### Run from a TOML configuration

```console
tud-lbm run examples/config_simple.toml
```

Use `--dry-run` to inspect a configuration without running it, and `--override` to change values from the command line:

```console
tud-lbm run examples/config_simple.toml --dry-run
tud-lbm run examples/config_simple.toml --override tau=0.8 --override nt=5_000
```

List the available physics, lattice, and analysis operators with:

```console
tud-lbm run --list-simulation-operators
tud-lbm run --list-simulation-analysis
```

### Interactive CLI

Omit the configuration path to enter interactive mode:

```console
tud-lbm run
```

### Parameter sweeps and post-processing

List-valued sweepable fields in a TOML configuration expand into a parameter sweep. Use `--max-workers` to control parallel execution and `--compare` to create cross-run comparison plots.

```console
tud-lbm run examples/config_parallel.toml --max-workers 4 --compare
```

Saved runs can be plotted or animated, and a directory of runs can be analysed:

```console
tud-lbm visualise /path/to/run-directory --no-prompt
tud-lbm animate /path/to/run-directory
tud-lbm compare /path/to/sweep-results --no-prompt
```

See the [examples](examples/) for single-phase, multiphase, parallel, and von Karman configurations.

## Contributing

If you want to contribute to the development of tud_lbm,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
