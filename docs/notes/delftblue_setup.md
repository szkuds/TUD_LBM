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
builds are separate packages. The scripted workflow (`scripts/README.md`)
installs **one build, the CUDA one**, and uses it for both kinds of node:

```bash
# on DelftBlue; scripts/db_pipeline.sh runs this for you
setup_on_delftblue.sh --release <sha12>
```

That runs `uv sync --extra cluster --frozen --no-install-project` into
`envs/<lock>/`, an environment shared by every commit with the same `uv.lock`,
and extracts the commit into `releases/<sha12>/` with its own `bin/tud-lbm`. A
GPU job uses the GPU; a CPU job exports `JAX_PLATFORMS=cpu`. Because releases
are never rewritten and there is only one build, CPU and GPU jobs can be queued
and running together, and a sync never disturbs them. The `cluster` extra
carries a `sys_platform == 'linux'` marker, so a plain `uv sync` on macOS
resolves to CPU JAX and never sees the CUDA packages.

Verify from a **GPU node**, not the login node — `jax.devices()` reports
`CpuDevice` on a login node no matter which build is installed:

```bash
srun --partition=<gpu-partition> --gpus-per-task=1 --time=00:05:00 \
     --account=<account> --pty \
     /scratch/<netid>/LBM/26_TUD_LBM/releases/current/.venv/bin/python -c 'import jax; print(jax.devices())'
```

To submit a GPU job, pass `--gpu` to `scripts/db_new_job.sh`. That overlays
`scripts/db_defaults.gpu.env` on the CPU defaults — partitions
`gpu-a100,gpu-v100` (whichever has room first), `gpus-per-task=1`, a small host
request (`cpus-per-task=4`, `mem-per-cpu=8GB`) and an 8 hour wall time — with
`--time` and the interactive prompts available on top. The partition
names and whether your account carries a GPU entitlement are cluster facts, not
repo facts; the values in that file were read off `sinfo -o "%20P %10G"` and
`sacctmgr show assoc user=$USER format=account,partition,qos` on 2026-08-31.

Two properties of the rendered job matter for the CUDA build specifically:

- **The job runs its own release, and only that.** The template puts
  `releases/<sha12>/bin` and that release's environment first on `PATH`;
  `UV_PROJECT_ENVIRONMENT` on its own is read by `uv` and puts nothing on
  `PATH`. No conda environment is activated and `MODULE_LOAD_LIST` is empty — uv
  ships its own Python.
- **A GPU job asserts its backend before doing any work.** The rendered script
  exits non-zero unless `jax.default_backend()` is `gpu`, because a CPU fallback
  otherwise completes normally and produces a plausible-looking result.

Do not add a CUDA module to `MODULE_LOAD_LIST`: the `jax[cuda12]` wheels bring
their own CUDA runtime via the `nvidia-*` packages and need only the node's
driver, and a system CUDA on `LD_LIBRARY_PATH` can shadow them.
