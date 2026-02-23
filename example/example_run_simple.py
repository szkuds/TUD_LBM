from config import configure_jax, SimulationBundle, SinglePhaseConfig, RunnerConfig

from core import Run
from util import visualise

# Configure JAX settings from central config
# To enable debugging (disable JIT), set DISABLE_JIT = True in config/jax_config.py
configure_jax()


def test_single_phase_simulation():
    """Test a single-phase LBM simulation."""
    print("\n=== Single-Phase LBM Simulation ===")

    # Create simulation bundle - modular configuration object
    bundle = SimulationBundle(
        simulation=SinglePhaseConfig(
            grid_shape=(100, 100),
            lattice_type="D2Q9",
            tau=0.6,
            nt=10000,
        ),
        runner=RunnerConfig(
            save_interval=1000,
            init_type="standard",
        ),
    )

    # Run simulation
    sim = Run(bundle)
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Single-Phase LBM Codebase")
    print("=" * 60)

    # Run simulation
    sim_single_phase_gravity = test_single_phase_simulation()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_single_phase_gravity, "Single-Phase")

    print("\nTest completed!")
