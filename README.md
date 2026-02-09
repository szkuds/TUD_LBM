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

## Contributing

If you want to contribute to the development of tud_lbm,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
