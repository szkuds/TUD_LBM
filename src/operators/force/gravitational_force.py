import jax.numpy as jnp
from .force import Force



class GravityForceMultiphase(Force):
    """
    Subclass for gravitational force, constant across the grid.
    """

    def __init__(
        self, force_g: float, inclination_angle_deg: float, grid_shape: tuple
    ):
        if grid_shape.__len__() != 2:
            raise ValueError("Currently supports 2D (d=2) only")

        self.name = 'GravitationalForce'

        force_x = force_g * -jnp.sin(jnp.deg2rad(inclination_angle_deg))
        force_y = force_g * jnp.cos(jnp.deg2rad(inclination_angle_deg))

        force_array = jnp.zeros((grid_shape[0], grid_shape[1], 1, grid_shape.__len__()))
        force_array = force_array.at[:, :, 0, 0].set(force_x)
        force_array = force_array.at[:, :, 0, 1].set(force_y)

        super().__init__(force_array)

    def compute_force(self, **kwargs) -> jnp.ndarray:
        rho = kwargs.get('rho')
        if rho is None:
            raise ValueError("GravityForceMultiphase requires 'rho' in kwargs")
        return -self.force * rho
