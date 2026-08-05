# OpenSSF Best Practices badge — answer sheet

Working document for maintainers. Not part of the user-facing documentation.

The badge lives at <https://www.bestpractices.dev/projects/11832>. It sat at **45%** because most
of the questionnaire had never been filled in — not because the criteria were failing. This file
holds the justification text to paste into each unanswered field.

**How to use:** log in to the badge page with GitHub, click *Edit*, and work through the sections
below. Each row gives the answer to select and the justification text. Keep this file up to date
when the answers stop being true.

Repository URLs used throughout:

- Repo: <https://github.com/szkuds/TUD_LBM>
- Docs: <https://tud-lbm.readthedocs.io/en/latest/>
- CI: <https://github.com/szkuds/TUD_LBM/actions>

---

## Requires your judgement — do not paste blindly

Four criteria depend on facts only you can confirm. Answer these yourself.

| Criterion | Why it needs you |
|---|---|
| `know_secure_design` | Asserts that at least one primary developer knows how to design secure software. Answer honestly; the badge asks for a self-assessment, and "Met" is reasonable if you understand least privilege, input validation, and the project's own threat model as written in `SECURITY.md`. |
| `know_common_errors` | Same, for common vulnerability classes (injection, buffer overflow, race conditions) and their mitigations. |
| `report_responses` | Requires that the majority of bug reports in the last 2–12 months got a response. Check the issue tracker before answering — there are open issues dating back to April 2026 (#53, #70, #82, #83). If several sat unanswered, answer "Unmet" and fix it by replying, rather than overstating. |
| `enhancement_responses` | Same, for enhancement requests. Note this one is SHOULD, not MUST, and an explicit "we do not commit to responding to all enhancement requests" is an acceptable answer. |

---

## Basics

| Criterion | Answer | Justification |
|---|---|---|
| `description_good` | Met | The README describes the project as a JAX-accelerated lattice Boltzmann method framework supporting single-phase, multiphase, wetting, hysteresis and chemical-step simulations: https://github.com/szkuds/TUD_LBM#readme |
| `interact` | Met | The README's "Reporting bugs and vulnerabilities" section links the issue tracker and SECURITY.md; CONTRIBUTING.md documents how to contribute. |
| `contribution` | Met | https://github.com/szkuds/TUD_LBM/blob/main/CONTRIBUTING.md |
| `contribution_requirements` | Met | CONTRIBUTING.md states the testing policy and the local quality gate (ruff format, ruff check, ty check, pytest) that a contribution must pass. |
| `floss_license` | Met | Apache License 2.0. |
| `floss_license_osi` | Met | Apache-2.0 is OSI-approved. |
| `license_location` | Met | https://github.com/szkuds/TUD_LBM/blob/main/LICENSE |
| `documentation_basics` | Met | https://tud-lbm.readthedocs.io/en/latest/ — quickstart, architecture and operator documentation. |
| `documentation_interface` | Met | The docs cover the Python API and the `tud-lbm` CLI; `docs/notes/README.dev.md` documents the operator registry and package reference. |
| `sites_https` | Met | Both github.com and readthedocs.io serve over HTTPS. |
| `discussion` | Met | GitHub Issues, which is searchable: https://github.com/szkuds/TUD_LBM/issues |
| `english` | Met | All documentation, code comments and issues are in English. |
| `maintained` | Met | Actively developed; commits and merged pull requests are continuous through 2026. |

## Change Control

| Criterion | Answer | Justification |
|---|---|---|
| `repo_public` | Met | https://github.com/szkuds/TUD_LBM |
| `repo_track` | Met | Git records author and timestamp for every change. |
| `repo_interim` | Met | Work is merged to `main` via pull requests continuously between releases. |
| `repo_distributed` | Met | Git. |
| `version_unique` | Met | Calendar versioning, `YYYY.MINOR.PATCH`; the current version is 2026.0.1, kept consistent across `pyproject.toml`, `src/__init__.py`, `CITATION.cff` and `docs/conf.py` by bump-my-version. |
| `version_semver` | Met | CalVer, which the criterion accepts alongside SemVer. |
| `version_tags` | Met | Releases are tagged: https://github.com/szkuds/TUD_LBM/tags |
| `release_notes` | Met | https://github.com/szkuds/TUD_LBM/releases and https://github.com/szkuds/TUD_LBM/blob/main/CHANGELOG.md |
| `release_notes_vulns` | Met | CHANGELOG.md reserves a "Security" heading per release for fixed vulnerabilities, with CVE identifiers where assigned; CONTRIBUTING.md makes recording them a release step. No vulnerabilities have been reported to date. |

## Reporting

| Criterion | Answer | Justification |
|---|---|---|
| `report_process` | Met | CONTRIBUTING.md documents how to file bugs and questions; the README links the tracker; issue templates prompt for the required detail: https://github.com/szkuds/TUD_LBM/blob/main/CONTRIBUTING.md |
| `report_tracker` | Met | https://github.com/szkuds/TUD_LBM/issues |
| `report_responses` | *see judgement section above* | |
| `enhancement_responses` | *see judgement section above* | |
| `report_archive` | Met | GitHub Issues is a publicly readable, permanent archive of reports and responses. |
| `vulnerability_report_process` | Met | https://github.com/szkuds/TUD_LBM/blob/main/SECURITY.md |
| `vulnerability_report_private` | Met | GitHub private vulnerability reporting (draft security advisories) is enabled, with email to the maintainer as a fallback. Both are documented in SECURITY.md. |
| `vulnerability_report_response` | Met | SECURITY.md commits to acknowledging reports within 14 days. No vulnerabilities have been reported to date. |

## Quality

| Criterion | Answer | Justification |
|---|---|---|
| `build` | Met | Standard Python build (`python -m build`) via a PEP 621 `pyproject.toml`; verified in CI: https://github.com/szkuds/TUD_LBM/blob/main/.github/workflows/build.yml |
| `build_common_tools` | Met | uv, setuptools via the standard PEP 517 `setuptools.build_meta` backend, and `python -m build` — all standard Python tooling. |
| `build_floss` | Met | Every build and test dependency (uv, CPython, JAX, NumPy, SciPy, pytest, ruff, ty) is FLOSS. |
| `test` | Met | pytest suite of 71 test modules under `tests/`, covering unit, integration and protocol-conformance levels. |
| `test_invocation` | Met | `uv run pytest` — the standard invocation for a Python project. |
| `test_most` | Met | Coverage is measured on every push and pull request and reported to SonarCloud: https://sonarcloud.io/dashboard?id=szkuds_TUD_LBM |
| `test_continuous_integration` | Met | GitHub Actions runs the full suite on every push to `main` and every pull request, across a matrix of 3 operating systems (Ubuntu, macOS, Windows) and 4 Python versions (3.11–3.14): https://github.com/szkuds/TUD_LBM/blob/main/.github/workflows/build.yml |
| `test_policy` | Met | CONTRIBUTING.md, "Testing policy": new operators, CLI commands, configuration fields and analysis routines must come with tests; bug fixes must come with a regression test; new code must not decrease coverage. |
| `tests_are_added` | Met | Recent pull requests add tests alongside functional changes; the SonarCloud quality gate enforces coverage on new code for every pull request. |
| `tests_documented_added` | Met | The testing policy is in CONTRIBUTING.md, which is the instructions for proposing changes. |
| `warnings` | Met | ruff with `select = ["ALL"]` (every rule enabled, suppressions narrowed per file) plus the ty type checker. Both run in CI and as pre-commit hooks. |
| `warnings_fixed` | Met | CI fails on any ruff or ty finding, so `main` is always clean. |
| `warnings_strict` | Met | `select = ["ALL"]` is ruff's maximally strict setting. Suppressions are per-file and enumerated in `pyproject.toml` rather than blanket-disabled. |

## Security

Most of this section is not applicable: TUD-LBM is an offline scientific simulation library that
performs no cryptography, has no network listener, and handles no credentials or personal data.
The threat model is documented in SECURITY.md.

| Criterion | Answer | Justification |
|---|---|---|
| `know_secure_design` | *see judgement section above* | |
| `know_common_errors` | *see judgement section above* | |
| `crypto_published` | N/A | The project implements no cryptographic functionality. |
| `crypto_call` | N/A | Same. |
| `crypto_floss` | N/A | Same. |
| `crypto_keylength` | N/A | Same. |
| `crypto_working` | N/A | Same. |
| `crypto_weaknesses` | N/A | Same. The only hash in the codebase is a SHA-256 digest used as a non-security cache key for surface-tension calibration data. |
| `crypto_pfs` | N/A | The project performs no key agreement. |
| `crypto_password_storage` | N/A | The project stores no passwords. |
| `crypto_random` | N/A | Random numbers are used only for physics initialisation via JAX PRNG, never for security. |
| `delivery_mitm` | Met | Distributed over HTTPS via PyPI and GitHub, and cloned over HTTPS or SSH. |
| `delivery_unsigned` | Met | No cryptographic hashes are retrieved over unprotected channels; uv verifies dependency hashes from `uv.lock`. |
| `vulnerabilities_fixed_60_days` | Met | No vulnerabilities have been reported. |
| `vulnerabilities_critical_fixed` | Met | No vulnerabilities have been reported. |
| `no_leaked_credentials` | Met | The repository contains no credentials. `.env.example` holds placeholder values only, and CI secrets are supplied through GitHub Actions secrets. |

## Analysis

| Criterion | Answer | Justification |
|---|---|---|
| `static_analysis` | Met | SonarCloud analyses the project on every push to `main` and every pull request: https://sonarcloud.io/dashboard?id=szkuds_TUD_LBM. ruff and ty additionally run in CI and as pre-commit hooks. |
| `static_analysis_common_vulnerabilities` | Met | SonarCloud includes security hotspot and vulnerability rules for Python; ruff's enabled rule set includes flake8-bandit (`S`) security checks. |
| `static_analysis_fixed` | Met | The SonarCloud quality gate must pass before a pull request is merged. |
| `static_analysis_often` | Met | On every push and every pull request, not merely before releases. |
| `dynamic_analysis` | Unmet | Not currently performed. Optional (SUGGESTED); the test suite exercises the full simulation pipeline, and Python is memory-safe. |
| `dynamic_analysis_unsafe` | N/A | Python is memory-safe; the project contains no C, C++, or other memory-unsafe code. |
| `dynamic_analysis_enable_assertions` | Met | Tests run with assertions enabled (pytest's default; the suite is assertion-based). |
| `dynamic_analysis_fixed` | Met | No vulnerabilities have been found through dynamic analysis. |

---

## One-time repository settings

These are clicks in the GitHub UI, not file changes, and the badge answers above depend on them:

1. **Settings → Code security → Private vulnerability reporting → Enable.** Without this the
   advisory link in SECURITY.md returns 404 and `vulnerability_report_private` genuinely fails.
2. Confirm **Settings → General → Features → Issues** stays enabled.
3. After `SECURITY.md` reaches `main`, check that GitHub surfaces it on the Security tab:
   `gh repo view --json securityPolicyUrl` should return a non-empty URL.
