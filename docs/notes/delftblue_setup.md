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
git archive --format=tar.gz --output=tud_lbm.tar.gz HEAD
```

`git archive HEAD` produces a clean snapshot of committed files only — no `.venv`, no `__pycache__`, no `.git` bloat.

---

## 2. Transfer to DelftBlue

```bash
scp tud_lbm.tar.gz <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/
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
export UV_PROJECT_ENVIRONMENT=/scratch/<netid>/tud_lbm/.venv
```

Then reload:

```bash
source ~/.bashrc
```

---

## 5. Extract and sync

```bash
cd /scratch/<netid>
mkdir tud_lbm && tar -xzf tud_lbm.tar.gz -C tud_lbm
cd tud_lbm

uv sync
```

`uv sync` installs the correct Python version and all dependencies from `uv.lock` in one step.

---

## Subsequent updates

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

---

## JAX CPU/GPU note

Login nodes have no GPU. DelftBlue GPU nodes use CUDA. JAX's CPU and GPU builds are separate, so verify your `pyproject.toml` pins the correct variant for the cluster environment before syncing.

Consider using a dependency group or environment marker to separate local (CPU) and cluster (CUDA) JAX installs if you run on both.
