"""Example: Parallel parameter sweep simulation.

Demonstrates how to:
1. Load a configuration with array parameters
2. Expand it into multiple SimulationConfig objects
3. Execute them in parallel
4. Save results manifest

Configuration is loaded from config_parallel.toml.
"""

import sys
from pathlib import Path
from src.config.adapter_toml import TomlAdapter
from src.config.array_expansion import enumerate_configs
from src.config.array_expansion import expand_config
from src.config.jax_config import configure_jax
from src.pipeline.parallel_runner import run_parallel_simulations
from src.pipeline.parallel_runner import save_sweep_log

# Configure JAX (64-bit precision, JIT enabled).
configure_jax()


def run_parallel_sweep():
    """Execute a parameter sweep with parallel simulations."""
    # Load configuration from TOML file (with array parameters preserved)
    config_path = Path(__file__).parent / "config_parallel.toml"
    adapter = TomlAdapter()
    config_dict = adapter.load_raw(str(config_path))

    # Detect and expand array parameters
    configs, metadata = expand_config(config_dict)

    if metadata:
        for _field, _values in metadata.array_values.items():
            pass
    else:
        pass

    # Optional: Enumerate and display each configuration before execution
    for _idx, params, _cfg in enumerate_configs(config_dict):
        (", ".join(f"{k}={v}" for k, v in params.items()) if params else "(base)")

    # Execute simulations in parallel

    # Extract parameters for each config
    params_list = [params for idx, params, cfg in enumerate_configs(config_dict)]

    results = run_parallel_simulations(
        configs,
        max_workers=4,
        parameters_list=params_list,
        verbose=True,
        continue_on_error=True,
    )

    # Save results manifest
    results_dir = Path(configs[0].results_dir).expanduser() / "sweep_manifest"
    save_sweep_log(results, results_dir)

    # Summary
    sum(1 for r in results if r.status == "success")
    sum(1 for r in results if r.status == "failed")

    return results


if __name__ == "__main__":
    results = run_parallel_sweep()
    sys.exit(0 if all(r.status == "success" for r in results) else 1)
