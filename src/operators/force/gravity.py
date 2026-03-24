"""Gravity force for multiphase simulations — pure functions.

Provides a setup-time builder and a jittable per-step computation.

Usage::

    # Setup time (once)
    template = build_gravity_force((64, 64), force_g=0.001)

    # Step time (every iteration, inside JIT)
    force = compute_gravity_force(template, rho)
"""

from __future__ import annotations
import jax.numpy as jnp
from registry import force_model


@force_model(name="gravity_multiphase")
def build_gravity_force(
    grid_shape: tuple[int, int],
    force_g: float,
    inclination_angle_deg: float = 0.0,
) -> jnp.ndarray:
    """Build a constant gravity-force template array.

    The template encodes the direction and magnitude of gravity.
    At each time step, the actual force is ``-template * rho``.

    Args:
        grid_shape: ``(nx, ny)`` spatial dimensions.
        force_g: Magnitude of gravitational acceleration.
        inclination_angle_deg: Inclination angle in degrees (0 = vertical).

    Returns:
        Gravity template, shape ``(nx, ny, 1, 2)``.
    """
    nx, ny = grid_shape[:2]
    angle_rad = jnp.deg2rad(inclination_angle_deg)
    force_x = force_g * (-jnp.sin(angle_rad))
    force_y = force_g * jnp.cos(angle_rad)

    force_array = jnp.zeros((nx, ny, 1, 2))
    force_array = force_array.at[:, :, 0, 0].set(force_x)
    return force_array.at[:, :, 0, 1].set(force_y)


def compute_gravity_force(
    gravity_template: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Compute per-step gravity force: ``F = -template * rho``.

    This function is fully jittable and should be called inside the
    ``lax.scan`` body.

    Args:
        gravity_template: Pre-computed template from
            :func:`build_gravity_force`, shape ``(nx, ny, 1, 2)``.
        rho: Density field, shape ``(nx, ny, 1, 1)``.

    Returns:
        Gravity force field, shape ``(nx, ny, 1, 2)``.
    """
    return -gravity_template * rho
