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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config.adapter_toml import TomlAdapter
from config.array_expansion import enumerate_configs, expand_config
from config.jax_config import configure_jax
from runner.parallel_runner import run_parallel_simulations, save_sweep_log

# Configure JAX (64-bit precision, JIT enabled).
configure_jax()


def run_parallel_sweep():
    """Execute a parameter sweep with parallel simulations."""
    print("\n=== Parallel Simulation Sweep ===")

    # Load configuration from TOML file (with array parameters preserved)
    config_path = Path(__file__).parent / "config_parallel.toml"
    adapter = TomlAdapter()
    config_dict = adapter.load_raw(str(config_path))

    print(f"✓ Config loaded from: {config_path.name}")
    print(f"  Raw config keys: {list(config_dict.keys())}")

    # Detect and expand array parameters
    configs, metadata = expand_config(config_dict)

    print("\n✓ Array expansion:")
    if metadata:
        print(f"  Array fields: {sorted(metadata.field_names)}")
        for field, values in metadata.array_values.items():
            print(f"    - {field}: {values}")
        print(f"  Total configurations: {metadata.total_combinations}")
    else:
        print("  No array parameters found (single configuration)")

    # Optional: Enumerate and display each configuration before execution
    print("\n✓ Configuration details:")
    for idx, params, cfg in enumerate_configs(config_dict):
        params_str = (
            ", ".join(f"{k}={v}" for k, v in params.items()) if params else "(base)"
        )
        print(f"  [{idx}] {params_str}")
        print(f"      grid: {cfg.grid_shape}, tau: {cfg.tau}, nt: {cfg.nt}")

    # Execute simulations in parallel
    print(f"\n✓ Starting parallel execution ({len(configs)} simulations)...")

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
    print(f"✓ Manifest saved to: {results_dir / 'sweep_manifest.json'}")

    # Summary
    successful = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    print(f"\n{'=' * 70}")
    print(f"Results: {successful} successful, {failed} failed")
    print(f"{'=' * 70}\n")

    return results


if __name__ == "__main__":
    results = run_parallel_sweep()
    sys.exit(0 if all(r.status == "success" for r in results) else 1)
