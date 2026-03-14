"""Multiphase LBM wetting/hysteresis simulation example.

Uses the streaming I/O path to write snapshots to disk during the
``jax.lax.scan`` loop via ``jax.debug.callback``.
"""

import os
import tempfile

from config.jax_config import configure_jax
from config.simulation_config import SimulationConfig
from setup.simulation_setup import build_setup
from runner.run import init_state, run

# Configure JAX (64-bit precision, JIT enabled).
configure_jax()


def wetting_hysteresis_simulation():
    """Run a multiphase wetting simulation with streaming I/O."""
    print("\n=== Multiphase Wetting with Streaming I/O ===")

    config = SimulationConfig(
        sim_type="multiphase",
        simulation_name="wetting_demo",
        grid_shape=(201, 101),
        lattice_type="D2Q9",
        tau=0.99,
        nt=2000,
        save_interval=100,
        init_type="wetting",
        eos="double-well",
        kappa=0.017,
        rho_l=1.0,
        rho_v=0.33,
        interface_width=4,
        force_enabled=True,
        # force_config is now a list of plain dicts — no class instantiation
        force_config=[
            {
                "type": "gravity_multiphase",
                "force_g": 2e-6,
                "inclination_angle_deg": 60,
            }
        ],
        bc_config={
            "left": "periodic",
            "bottom": "wetting",
            "top": "symmetry",
            "right": "periodic",
            "wetting_params": {
                "phi_left": 1.0,
                "phi_right": 1.0,
                "d_rho_left": 0.0,
                "d_rho_right": 0.0,
            },
        },
        hysteresis_config={
            "ca_advancing": 90.0,
            "ca_receding": 80.0,
            "learning_rate": 0.05,
            "max_iterations": 10,
        },
        save_fields=["rho", "u"],
        results_dir=os.path.join(tempfile.gettempdir(), "tud_lbm_demo"),
    )

    setup = build_setup(config)
    # init_state now uses setup.init_type ("wetting") via the
    # operators.initialise factory to produce a proper sessile-droplet f
    state = init_state(setup)

    print(f"  Setup complete — init_type={setup.init_type}")
    print(f"  Gravity template present: {setup.gravity_template is not None}")

    # In-memory trajectory (no io_handler for this demo)
    final_state, trajectory = run(
        setup,
        state,
        nt=config.nt,
        save_interval=config.save_interval,
    )

    print(f"  Final time step : {int(final_state.t)}")
    if trajectory is not None:
        print(f"  Trajectory snaps: {trajectory.f.shape[0]}")
    return final_state


if __name__ == "__main__":
    print("TUD-LBM  —  Multiphase Wetting Example")
    print("=" * 50)

    wetting_hysteresis_simulation()

    print("\nDone.")
