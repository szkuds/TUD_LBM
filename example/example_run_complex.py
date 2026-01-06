import jax

from src.core import Run
from src.operators import GravityForceMultiphase
from src.util import visualise

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


def wetting_hysteresis_simulation_test():
    """Test LBM wetting implementation with hysteresis enabled."""
    print("\n=== Testing LBM Wetting with Hysteresis ===")

    # Simulation parameters
    grid_shape = (401, 101)
    tau = 0.99
    nt = 20000
    save_interval = 2000
    kappa = 0.017
    rho_l = 1.0
    rho_v = 0.33
    interface_width = 4

    phi_value = 1.0
    d_rho_value = 0.0

    force_g = 2e-6
    inclination_angle = 60
    gravity = GravityForceMultiphase(
        force_g, inclination_angle, grid_shape
    )

    # Add hysteresis parameters to bc_config
    bc_config = {
        'left': 'periodic',
        'bottom': 'wetting',
        'top': 'symmetry',
        'right': 'periodic',
        'wetting_params': {
            'rho_l': rho_l,
            'rho_v': rho_v,
            'phi_left': phi_value,
            'phi_right': phi_value,
            'd_rho_left': d_rho_value,
            'd_rho_right': d_rho_value,
            'width': interface_width
        },
        'hysteresis_params': {
            'ca_advancing': 90.0,
            'ca_receding': 80.0,
            'learning_rate': 0.05,
            'max_iterations': 10
        }
    }

    sim = Run(
        simulation_type="multiphase",
        grid_shape=grid_shape,
        lattice_type="D2Q9",
        tau=tau,
        nt=nt,
        kappa=kappa,
        rho_l=rho_l,
        rho_v=rho_v,
        interface_width=interface_width,
        save_interval=save_interval,
        bc_config=bc_config,
        force_enabled=True,
        force_obj=[gravity],
        phi_value=phi_value,
        d_rho_value=d_rho_value,
        wetting_enabled=True,
        hysteresis_params=bc_config['hysteresis_params'],
        init_type="wetting",
    )

    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    sim_wetting_hysteresis = wetting_hysteresis_simulation_test()

    # Visualize results
    print("\n=== Visualizing Wetting Hysteresis Test Results ===")
    visualise(sim_wetting_hysteresis, "Wetting Hysteresis Implementation Test")

    print("\nTest completed! Check the 'results' directory for data and plots.")
