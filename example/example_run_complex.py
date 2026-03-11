from app_setup import configure_jax, SimulationSetup

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

    setup = SimulationSetup(
        sim_type="multiphase",
        grid_shape=(201, 101),
        lattice_type="D2Q9",
        tau=0.99,
        nt=2000,
        eos="double-well",
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
        save_interval=200,
        init_type="wetting",
    )

    # Run simulation
    sim = Run(setup)
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Multiphase LBM Codebase with Wetting Hysteresis")
    print("=" * 60)

    # Run simulation
    sim_wetting_hysteresis = wetting_hysteresis_simulation_test()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_wetting_hysteresis, "Wetting Hysteresis Implementation Test")

    print("\nTest completed!")
