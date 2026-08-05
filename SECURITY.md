# Security Policy

## Supported versions

Security fixes are applied to the most recent release line only. TUD-LBM uses calendar versioning
(the current version is `2026.0.1`); if you are running an older release, please upgrade before
reporting an issue.

| Version   | Supported          |
|-----------|--------------------|
| `2026.0.x`| :white_check_mark: |
| `< 2026.0`| :x:                |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report privately through one of these channels:

1. **GitHub private vulnerability reporting** (preferred) —
   [open a draft security advisory](https://github.com/szkuds/TUD_LBM/security/advisories/new).
   This keeps the report visible only to you and the maintainers until a fix is published.
2. **Email** — `szkudx@gmail.com`, with `[TUD-LBM security]` in the subject line.

Please include, as far as you can determine them:

- the affected version or commit SHA;
- a description of the vulnerability and its impact;
- steps to reproduce, ideally with a minimal configuration file or script;
- your operating system and Python version.

## What to expect

- We will **acknowledge your report within 14 days**.
- If the report is accepted, we will keep you informed of progress toward a fix and let you know
  when it is released.
- If the report is declined, we will explain why.
- Fixes are published as a new release, and the vulnerability is recorded in
  [`CHANGELOG.md`](CHANGELOG.md) and in a GitHub security advisory.
- We are happy to credit reporters in the advisory unless you ask us not to.

## Scope and threat model

TUD-LBM is an offline scientific simulation library. Being explicit about what it is not helps you
judge whether a finding is in scope:

- It **does not** listen on a network socket, serve HTTP, or expose a remote API.
- It **does not** perform authentication, authorisation, or session management.
- It **does not** implement or configure cryptography. The only hash in the codebase is a
  non-security SHA-256 digest used as a cache key for surface-tension calibration data
  (`src/simulation_io/analysis/surface_tension/surface_tension.py`).
- It **does not** handle personal data or secrets. `.env.example` contains placeholders only.

The realistic attack surface is **untrusted input files**, and the security assumption is that you
supply your own:

- **TOML configuration files** parsed by `src/config/adapter_toml.py`. Configurations select
  operators by name and set numeric parameters; they are not executed as code. Running a
  configuration you did not write is roughly as risky as running any third-party script — inspect
  it first.
- **`.npz`/`.npy` snapshot files** read by the analysis and visualisation code. These are loaded
  with `allow_pickle` left at the NumPy default of `False`; do not change that when reading files
  from an untrusted source, as pickle deserialisation permits arbitrary code execution.
- **Parallel sweeps** (`src/pipeline/parallel_runner.py`) pickle work functions to send them to a
  local `ProcessPoolExecutor`. This is process-local and never reads pickles from disk or network.

Findings in these areas are in scope. Reports that require an attacker to already control the
machine running the simulation, or that consist solely of a numerically unstable configuration
producing `NaN`, generally are not — please file the latter as a normal bug report.
