## Badges

(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations |                                                                                                                                                                                                                                                  |
| :-- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/szkuds/tud_lbm)                                                                                               |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/szkuds/tud_lbm)](https://github.com/szkuds/TUD_LBM?tab=Apache-2.0-1-ov-file)                                                                                                      |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-tud_lbm-00a3e3.svg)](https://research-software-directory.org/software/tudlbm) [![workflow pypi badge](https://img.shields.io/pypi/v/tud_lbm.svg?colorB=blue)](https://pypi.python.org/project/tud_lbm/) |
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>)                                                                                                                                |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/11832/badge)](https://bestpractices.coreinfrastructure.org/projects/11832)                                                                                          |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)                                                                  |
| **Other best practices**           | &nbsp;                                                                                                                                                                                                                                           |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=szkuds_TUD_LBM&metric=alert_status)](https://sonarcloud.io/dashboard?id=szkuds_TUD_LBM)                                                                          |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=szkuds_TUD_LBM&metric=coverage)](https://sonarcloud.io/dashboard?id=szkuds_TUD_LBM)                                                                              || Documentation                      | [![Documentation Status](https://readthedocs.org/projects/tud_lbm/badge/?version=latest)](https://tud_lbm.readthedocs.io/en/latest/?badge=latest) || **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](https://github.com/szkuds/tud_lbm/actions/workflows/build.yml/badge.svg)](https://github.com/szkuds/tud_lbm/actions/workflows/build.yml)                                                                                               |
| Citation data consistency          | [![cffconvert](https://github.com/szkuds/tud_lbm/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/szkuds/tud_lbm/actions/workflows/cffconvert.yml)                                                                                || SonarCloud                         | [![sonarcloud](https://github.com/szkuds/tud_lbm/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/szkuds/tud_lbm/actions/workflows/sonarcloud.yml) |## How to use tud_lbm

This is a lattice Boltzmann method package developed at Delft University of Technology.

The project setup is documented in [project_setup.md](project_setup.md). Feel free to remove this document (and/or the link to this document) if you don't need it.

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

#### Installing Optional Dependencies

To use hysteresis optimization features (requires `optax`):

```console
conda install -c conda-forge optax
```

Or install with the optional dependency group:

```console
pip install -e ".[hysteresis]"
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

## Testing

Verify your installation by running the test suite. This ensures all dependencies are correctly installed and the package works as expected.

### Running Tests with Conda

```console
# Activate the environment (if not already activated)
conda activate tud-lbm

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_optional_dependencies.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Running Tests with conda run

If you prefer not to activate the environment:

```console
# Run all tests
conda run -n tud-lbm pytest tests/ -v

# Run with verbose output and short traceback
conda run -n tud-lbm pytest tests/ -v --tb=short
```

## Documentation

Include a link to your project's full documentation here.

## Contributing

If you want to contribute to the development of tud_lbm,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
