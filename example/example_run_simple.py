import numpy as np
from src.core import Run
from src.util import visualise
import jax

# this line is added for debugging
# jax.config.update("jax_disable_jit", True)
# Force double precision
jax.config.update("jax_enable_x64", True)



def test_single_phase_simulation():
    """Test a single-phase LBM simulation with gravity."""
    print("\n=== Single-Phase LBM Simulation ===")

    grid_shape = (100, 100)
    tau = .6
    nt = 10000
    save_interval = 1000

    sim = Run(
        simulation_type="singlephase",
        grid_shape=grid_shape,
        lattice_type="D2Q9",
        tau=tau,
        nt=nt,
        save_interval=save_interval,
        init_type="standard"
    )
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
