"""Single-phase LBM simulation example using the functional API.

Demonstrates the in-memory trajectory execution mode — short runs where
the full trajectory fits in device memory.
"""

from config.jax_config import configure_jax
from config.simulation_config import SimulationConfig
from setup.simulation_setup import build_setup
from runner.run import init_state, run

# Configure JAX (64-bit precision, JIT enabled).
configure_jax()


def run_in_memory():
    """Run a short simulation and keep the trajectory in memory."""
    print("\n=== In-Memory Trajectory Mode ===")

    config = SimulationConfig(
        grid_shape=(100, 100),
        lattice_type="D2Q9",
        tau=0.6,
        nt=100,
        save_interval=10,
        init_type="standard",
    )

    setup = build_setup(config)
    # init_state now uses setup.init_type ("standard") via the
    # operators.initialise factory
    state = init_state(setup)

    print(f"  Init type       : {setup.init_type}")
    print(f"  Initial f shape : {state.f.shape}")

    # No io_handler → full trajectory returned
    final_state, trajectory = run(
        setup,
        state,
        nt=config.nt,
        save_interval=config.save_interval,
    )

    print(f"  Final time step : {int(final_state.t)}")
    print(f"  Trajectory shape: {trajectory.f.shape[0]} snapshots")
    return final_state, trajectory


if __name__ == "__main__":
    print("TUD-LBM  —  Single-Phase Example")
    print("=" * 50)

    run_in_memory()

    print("\nDone.")
