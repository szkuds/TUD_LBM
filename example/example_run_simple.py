from app_setup import configure_jax, SimulationBundle, SinglePhaseConfig, RunnerConfig

from runner import Run
from util import visualise

# Configure JAX settings from central app_setup
# To enable debugging (disable JIT), set DISABLE_JIT = True in app_setup/jax_config.py
configure_jax()


def test_single_phase_simulation():
    """Test a single-phase LBM simulation_type."""
    print("\n=== Single-Phase LBM Simulation ===")

    # Create simulation_type bundle - modular configuration object
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

    # Run simulation_type
    sim = Run(bundle)
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Single-Phase LBM Codebase")
    print("=" * 60)

    # Run simulation_type
    sim_single_phase_gravity = test_single_phase_simulation()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_single_phase_gravity, "Single-Phase")

    print("\nTest completed!")
