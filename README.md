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

A lattice Boltzmann method package developed at Delft University of Technology. This package provides tools for simulating fluid dynamics using the lattice Boltzmann method, supporting both single-phase and multi-phase simulations.

The project setup is documented in [project_setup.md](project_setup.md).

## Installation

TUD-LBM requires Python 3.10 or newer.

We recommend using **uv** for installation because it provides a fast, consistent, and cross-platform workflow for Linux, macOS, and Windows. For most users, uv is the simplest way to create an isolated environment and install the package from the repository.

If you already work in a conda-based scientific Python environment, on an HPC system, or need non-Python/system-level dependencies managed outside the project, conda remains a good optional alternative.

### Install uv

Install `uv` by following the official instructions for your platform:

- https://docs.astral.sh/uv/getting-started/installation/

### Linux and macOS

```console
git clone git@github.com:szkuds/tud_lbm.git
cd tud_lbm
uv venv --python 3.14
source .venv/bin/activate
uv pip install .
python -c "import tud_lbm"
```

### Windows

```console
git clone git@github.com:szkuds/tud_lbm.git
cd tud_lbm
uv venv --python 3.14
.venv\Scripts\activate
uv pip install .
python -c "import tud_lbm"
```

### Optional: editable install for contributors

```console
uv pip install -e .[dev]
```

### Optional: conda

If you prefer conda for scientific Python environments, you can still use the conda-based workflow documented in [project_setup.md](project_setup.md).

## Documentation

For full documentation, visit [tud-lbm.readthedocs.io](https://tud-lbm.readthedocs.io/en/latest/).

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

For the full package reference (module-by-module API, operator tables, and architecture details), see the [developer documentation](README.dev.md#package-reference).

## Contributing

If you want to contribute to the development of tud_lbm,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
