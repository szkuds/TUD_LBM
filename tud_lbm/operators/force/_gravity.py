"""Gravity force module.

Provides a constant-body-force implementation with no auxiliary state.

Usage::

    # Via registry (preferred)
    from operators.force import build_force_fn

    module = build_force_fn("gravity_force")
    template = module.build({"force_g": 0.001}, (64, 64), config, lattice)
    force = module.compute(state, template)

    # Direct (internal / testing)
    from operators.force._gravity import GravityForceModule

    template = GravityForceModule.build({"force_g": 0.001}, (64, 64), config, lattice)
    force = GravityForceModule.compute(state, template)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from tud_lbm.registry import force_model

if TYPE_CHECKING:
    from tud_lbm.pipeline.state import State

# ══════════════════════════════════════════════════════════════════════
# ForceOperator protocol — registry-backed module
# ══════════════════════════════════════════════════════════════════════


@force_model(name="gravity_force")
class GravityForceModule:
    """Gravity force conforming to :class:`ForceOperator` protocol.

    Stateless — it uses the default no force state hooks.
    """

    @staticmethod
    def build(
        params: dict,
        grid_shape: tuple[int, ...],
        **_kwargs: object,
    ) -> jnp.ndarray:
        """Build a constant gravity-force template.

        Args:
            params: Config dict from ``[gravity_force]`` TOML section.
                Required key: ``force_g``.
                Optional key: ``inclination_angle_deg`` (default 0).
            grid_shape: Spatial dimensions ``(nx, ny, nz, ...)``.
            **kwargs: Additional arguments (config, lattice) ignored for stateless forces.

        Returns:
            Gravity template array, shape ``(nx, ny, nz, 1, d)``.
        """
        nx, ny, nz = grid_shape[:3]
        angle_rad = jnp.deg2rad(params.get("inclination_angle_deg", 0.0))
        force_x = params["force_g"] * (-jnp.sin(angle_rad))
        force_y = params["force_g"] * jnp.cos(angle_rad)

        template = jnp.zeros((nx, ny, nz, 1, 2))
        template = template.at[:, :, :, 0, 0].set(force_x)
        return template.at[:, :, :, 0, 1].set(force_y)

    @staticmethod
    def compute(
        state: State,
        precomputed: jnp.ndarray,
        **_kwargs: object,
    ) -> jnp.ndarray:
        """Compute gravity force (step-time, jittable).

        Args:
            state: Current simulation :class:`State`. Only ``state.f``
                is used (to compute density).
            precomputed: Gravity template from :meth:`build`.
            **kwargs: Additional arguments (ignored).

        Returns:
            Gravity force field, shape ``(nx, ny, nz, 1, d)``.
        """
        rho = jnp.sum(state.f, axis=-2, keepdims=True)
        return -precomputed * rho
