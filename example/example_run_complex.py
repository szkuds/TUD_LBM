from app_setup import configure_jax, SimulationSetup, MultiphaseConfig, RunnerConfig

from runner import Run
from simulation_operators import GravityForceMultiphase
from util import visualise

# Configure JAX settings from central app_setup
# To enable debugging (disable JIT), set DISABLE_JIT = True in app_setup/jax_config.py
configure_jax()


def wetting_hysteresis_simulation_test():
    """Test LBM wetting implementation with hysteresis enabled."""
    print("\n=== Testing LBM Wetting with Hysteresis ===")

    phi_value = 1.0
    d_rho_value = 0.0

    gravity = GravityForceMultiphase(
        force_g=2e-6,
        inclination_angle_deg=60,
        grid_shape=(201, 101),
    )

    # Create simulation_type bundle - modular configuration object
    bundle = SimulationBundle(
        simulation=MultiphaseConfig(
            grid_shape=(201, 101),
            lattice_type="D2Q9",
            tau=0.99,
            nt=2000,
            kappa=0.017,
            rho_l=1.0,
            rho_v=0.33,
            interface_width=4,
            force_enabled=True,
            force_obj=[gravity],
            bc_config={
                "left": "periodic",
                "bottom": "wetting",
                "top": "symmetry",
                "right": "periodic",
                "wetting_params": {
                    "phi_left": phi_value,
                    "phi_right": phi_value,
                    "d_rho_left": d_rho_value,
                    "d_rho_right": d_rho_value,
                },
                "hysteresis_params": {
                    "ca_advancing": 90.0,
                    "ca_receding": 80.0,
                    "learning_rate": 0.05,
                    "max_iterations": 10,
                },
            },
            extra={
                "phi_value": phi_value,
                "d_rho_value": d_rho_value,
                "wetting_enabled": True,
                "hysteresis_params": {
                    "ca_advancing": 90.0,
                    "ca_receding": 80.0,
                    "learning_rate": 0.05,
                    "max_iterations": 10,
                },
            },
        ),
        runner=RunnerConfig(
            save_interval=200,
            init_type="wetting",
        ),
    )

    # Run simulation_type
    sim = Run(bundle)
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Multiphase LBM Codebase with Wetting Hysteresis")
    print("=" * 60)

    # Run simulation_type
    sim_wetting_hysteresis = wetting_hysteresis_simulation_test()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_wetting_hysteresis, "Wetting Hysteresis Implementation Test")

    print("\nTest completed!")
