# TO DO

- **Debug flags don't reach sweep workers**: `--debug-stability` (like `--debug-wetting`) sets a module-level global in `config_overview`, which does not propagate to `ProcessPoolExecutor` sweep workers (spawn start method). Per-leg stability logging in parameter sweeps would need an env-var pass-through (e.g. checked in `configure_jax()`) later.
