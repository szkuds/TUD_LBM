"""Multiphase LBM wetting/hysteresis simulation example.

Uses the streaming I/O path to write snapshots to disk during the
``jax.lax.scan`` loop via ``jax.debug.callback``, then plots them
post-run using the registered plot operators.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.jax_config import configure_jax
from config.simulation_config import SimulationConfig
from setup.simulation_setup import build_setup
from runner.run import init_state, run
from util.io import SimulationIO
from util.plotting import FigureBuilder

# Configure JAX (64-bit precision, JIT enabled).
configure_jax()


def wetting_hysteresis_simulation():
    """Run a multiphase wetting simulation with streaming I/O + plotting."""
    print("\n=== Multiphase Wetting — Streaming I/O + Plotting ===")

    config = SimulationConfig(
        sim_type="multiphase",
        simulation_name="wetting_demo",
        grid_shape=(201, 101),
        lattice_type="D2Q9",
        tau=0.99,
        nt=2000,
        save_interval=200,
        init_type="wetting",
        eos="double-well",
        kappa=0.017,
        rho_l=1.0,
        rho_v=0.33,
        interface_width=4,
        save_fields=["rho", "u", "force"],
        plot_fields=["density", "velocity", "force", "analysis"],
        force_enabled=True,
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
        results_dir="~/TUD_LBM_data/results",
    )

    setup = build_setup(config)
    state = init_state(setup)

    print(f"  Grid            : {config.grid_shape}")
    print(f"  Steps           : {config.nt}  (save every {config.save_interval})")
    print(f"  Gravity present : {setup.gravity_template is not None}")

    # Create the I/O handler — makes the timestamped run directory.
    io = SimulationIO(
        base_dir=config.results_dir,
        config=config.to_dict(),
        simulation_name=config.simulation_name,
    )
    print(f"  Run directory   : {io.run_dir}")

    # Stream snapshots to disk during the lax.scan loop.
    final_state, _ = run(
        setup,
        state,
        nt=config.nt,
        save_interval=config.save_interval,
        io_handler=io,
        save_fields=tuple(config.save_fields),
    )

    print(f"  Final timestep  : {int(final_state.t)}")
    print(f"  Snapshots saved to: {io.data_dir}")

    # Render one composite PNG per saved snapshot.
    builder = FigureBuilder(config=config.to_dict(), run_dir=io.run_dir)
    saved_plots = builder.build_all()
    print(f"  Plots saved     : {len(saved_plots)} PNG(s) in {io.run_dir}/plots/")

    return final_state


if __name__ == "__main__":
    print("TUD-LBM  —  Multiphase Wetting Example")
    print("=" * 50)

    wetting_hysteresis_simulation()

    print("\nDone.")
