# TUD-LBM on DelftBlue — Setup Guide

## Overview

Since `git clone` is unavailable on DelftBlue login nodes, the workflow is:
**local archive → scp transfer → uv sync on cluster.**

---

## Prerequisites

Ensure the following files are committed in your local repo before archiving:

- `pyproject.toml`
- `uv.lock`
- `.python-version`

---

## 1. Local: create a clean archive

From your repo root:

```bash
git archive --format=tar.gz --output=src.tar.gz HEAD
```

`git archive HEAD` produces a clean snapshot of committed files only — no `.venv`, no `__pycache__`, no `.git` bloat.

---

## 2. Transfer to DelftBlue

```bash
scp src.tar.gz <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/
```

> **Use `/scratch`, not `/home`.** Package installs generate many small files and `/home` quota is limited.

---

## 3. On DelftBlue: install uv (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or follow the post-install instructions printed by the installer
```

---

## 4. Redirect uv cache to scratch (once)

Add these to your `~/.bashrc` to avoid filling `/home`:

```bash
export UV_CACHE_DIR=/scratch/<netid>/.uv_cache
export UV_PROJECT_ENVIRONMENT=/scratch/<netid>/src/.venv
```

Then reload:

```bash
source ~/.bashrc
```

---

## 5. Extract and sync

```bash
cd /scratch/<netid>
mkdir src && tar -xzf src.tar.gz -C src
cd src

uv sync
```

`uv sync` installs the correct Python version and all dependencies from `uv.lock` in one step.

---

## Subsequent updates

After making changes locally:

```bash
# Local
git archive --format=tar.gz --output=src.tar.gz HEAD
scp src.tar.gz <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/

# On DelftBlue
cd /scratch/<netid>/src
tar -xzf ../src.tar.gz   # overwrites changed files in place
uv sync                        # no-op if uv.lock hasn't changed
```

---

## JAX CPU/GPU note

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
