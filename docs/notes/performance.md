# Performance baseline: CPU vs GPU

Measured with `tud-lbm benchmark` (see `src/cli/benchmarking.py`). Every number
here is **float64** — `ENABLE_X64 = True` in `src/config/config_overview.py` is a
physics requirement for this project, not a tuning knob.

> **Read the fp64 caveat before quoting any figure.** On an A100, fp64 is half of
> fp32 arithmetic throughput and doubles memory traffic on a bandwidth-bound
> kernel. These numbers are a floor on what the hardware can do, not a measure of
> it. They are the right numbers *for this project*; they are not "the GPU speedup
> for LBM".

## How to reproduce

```bash
# Case A — bandwidth-bound LBM kernel. Sweep the grid to find the crossover.
tud-lbm benchmark examples/benchmarks/bench_kernel.toml \
    --steps 200 --repeats 3 --override 'grid_shape=[512,512]' --label kernel_512

# Case B — the real research workload, with the ablation ladder.
tud-lbm benchmark examples/benchmarks/bench_hysteresis.toml \
    --steps 20 --repeats 3 --breakdown --label hyst_256

# Price the streaming-I/O host callback (expected to matter far more on GPU).
tud-lbm benchmark examples/benchmarks/bench_kernel.toml --steps 200 --io
```

Each run writes a JSON record to `$TUD_LBM_DATA_DIR/benchmarks/<label>_<backend>.json`
carrying the backend, device kind, JAX version and x64 flag — that provenance is
what makes a CPU record and a GPU record comparable after the fact.

## Baseline finding: the hysteresis optimiser is the workload

Local CPU (Apple Silicon, jax 0.10.0, fp64), `multiphase_hysteresis` at 128×32,
`max_iterations = 10`, `trial_steps = 2`, measured from the initial state.

**Two numbers, and the difference between them is the point.**

`loss_tol = 0.0` — the optimiser always runs its full 10 iterations. This is the
fixed-work configuration `bench_hysteresis.toml` ships with, and the only one in
which a CPU number and a GPU number are comparable:

| Variant | ms/step | share of full |
|---|---|---|
| full | 87.6 | — |
| `no_optimiser` (`wetting_fn = None`) | 0.330 | optimiser = **99.6%** |
| `plain_step` (plain `multiphase` step) | 0.320 | CA/CLL measurement ≈ 0% |

`loss_tol = 1e-4` (the code default, `--override hysteresis.loss_tol=1e-4`) —
the `while_loop` exits early once the loss converges:

| Variant | ms/step | share of full |
|---|---|---|
| full | 1.42 | — |
| `no_optimiser` | 0.318 | optimiser = **77.5%** |
| `plain_step` | 0.321 | CA/CLL measurement ≈ 0% |

So the optimiser multiplier over a bare multiphase step is **~4.4× at best and
~274× at worst**, and which end you land on is decided entirely by how many Adam
iterations the early exit actually saves.

**Do not read the 4.4× as the production figure.** `_measure` restarts every
repeat from the *same* state, so with `loss_tol > 0` it prices the optimiser at
that one state — here a freshly initialised droplet sitting near its target
contact angle, where the loss is already small and the loop exits after roughly
one iteration. On a real trajectory, with the contact line moving under gravity,
the loss is larger and the iteration count climbs toward the cap. Treat 4.4× as a
floor, 274× as a ceiling, and measure the middle by instrumenting a real run
(`--debug-wetting` reports the actual `iters` against the cap in its per-side
row; note each logged step evaluates two extra objectives, so use it for
iteration counts and never for timing — and rate-limit it with
`--debug-wetting-interval` to keep even that cost off most steps).

Either way the conclusion for GPU work is the same: **the LBM kernel is not the
workload.** Even in the cheapest regime, collision, streaming and the
differential operators together account for under a quarter of the step, and the
contact-angle/contact-line measurement is unmeasurable at ~0%. Effort spent
optimising them is effort spent on a rounding error.

XLA's own cost model puts both cases at low arithmetic intensity — 0.21 FLOP/byte
for Case A, 0.14 for Case B — i.e. firmly **bandwidth-bound**. That is the profile
a GPU rewards, and it is the strongest a-priori argument for trying one.

## Results

_To be filled in from Phase 4. One row per (case, grid, backend)._

| Case | Grid | Backend | ms/step | MLUPS | compile (s) |
|---|---|---|---|---|---|
| | | | | | |

### Case A crossover

_The grid size at which the GPU overtakes the CPU. Below it, the GPU is
latency-bound on kernel launches and the CPU may genuinely win._

### Case B speedup

_Expect a different answer from Case A: the optimiser is a long serial chain of
data-dependent `while_loop` iterations, which is a weaker GPU shape than Case A's
dense stencil._

## Known measurement hazards

- **`while_loop` trip counts are data-dependent.** `hysteresis.py`'s `cond_fn`
  exits on `loss <= loss_tol`, so per-step cost varies with the physics state.
  `bench_hysteresis.toml` pins `loss_tol = 0.0` to force exactly `max_iterations`
  every step; without that, two backends starting from different states do
  different amounts of work and the comparison is meaningless.
- **Repeats do not chain.** `_measure` restarts every repeat from the same initial
  state, so the spread column is noise rather than physics. The corollary bites
  with `loss_tol > 0`: the harness then measures the optimiser's cost *at the
  initial state only*, which understates a real trajectory. This is the second
  reason `loss_tol = 0.0` is the primary configuration.
- **`--io` writes real snapshots** into a `[benchmark-io]` run directory under
  `results_dir`, with `save_interval` clamped to `--steps` so at least one write
  happens.
- **Compile time is a single up-front cost** for the whole `nt`-step loop, so it
  is reported separately and must not be amortised into ms/step.

## Deferred fixes

Measured by the harness, not fixed. Each is likely to hurt GPU more than CPU:

- `jax.debug.callback(..., ordered=True)` in `simulation_io/callbacks.py` threads a
  token through *every* scan iteration, whether or not that step saves, and
  `_host_save` does a full device-to-host copy plus a NaN scan of every field.
  `--io` prices this.
- `jax.lax.scan(..., jnp.arange(nt))` in `pipeline/runner.py` materialises an
  `nt`-length device array used only as a discarded loop counter; `length=nt`
  would do.
- `ProcessPoolExecutor` sweeps in `pipeline/parallel_runner.py` would put N
  processes on one GPU.
- Sharding a single run across devices (`pmap`/`shard_map`) — gated on this
  profiling, per `.claude/roadmap.md`.
