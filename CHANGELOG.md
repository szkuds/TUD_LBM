# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
uses [calendar versioning](https://calver.org/) in the form `YYYY.MINOR.PATCH`.

Security fixes are called out under a **Security** heading, with CVE identifiers where one was
assigned. See [SECURITY.md](SECURITY.md) for how to report a vulnerability.

## [Unreleased]

### Added

- `SECURITY.md` documenting the vulnerability reporting process, response commitment, and the
  project's threat model.
- GitHub issue templates for bug reports and feature requests.
- Explicit testing policy and local quality gate in `CONTRIBUTING.md`.
- This changelog.

### Fixed

- Version drift: `src/__init__.py`, `docs/conf.py` and `CITATION.cff` still reported `0.3.0` while
  `pyproject.toml` reported `2026.0.1`. All four now agree, which also unbreaks `bump-my-version`.
- Removed `[tool.ruff.lint.per-file-ignores]` entries for files deleted during the
  `src/util` → `src/simulation_io` and `src/runner` → `src/pipeline` reorganisation.
- The `ReadTheDocs` project URL used an underscore (`tud_lbm.readthedocs.io`), which is not a
  valid hostname and returned HTTP 400. Renamed to `Documentation` and pointed at
  `https://tud-lbm.readthedocs.io`.
- `pre-commit` and `bump-my-version` were required by `.pre-commit-config.yaml` and the release
  procedure but were not declared as development dependencies.

## 2026.0.1 — unreleased

Calendar versioning adopted; the version scheme changed from `0.x.y` to `YYYY.MINOR.PATCH`. This
version is set in `pyproject.toml` but has not been tagged or published as a GitHub release.

## [0.3.0] — 2026-07-30

### Added

- 3D lattice support (D3Q19); the core was converted to a 3D-normalised grid.
- Chemical-step hysteresis simulation type.
- Multi-input configuration saving.

### Changed

- Physics-first folder structure.
- Step function, hysteresis, and initialisation refactored.
- `README.md` reworked around `uv`.

### Fixed

- Wetting boundary condition behaviour.

## [0.0.3] — 2026-03-24

Initial alpha release.

[Unreleased]: https://github.com/szkuds/TUD_LBM/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/szkuds/TUD_LBM/compare/v0.0.3...v0.3.0
[0.0.3]: https://github.com/szkuds/TUD_LBM/releases/tag/v0.0.3
