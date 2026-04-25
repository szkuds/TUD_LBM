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
        **kwargs: object,
    ) -> jnp.ndarray:
        """Build a constant gravity-force template.

        Args:
            params: Config dict from ``[gravity_force]`` TOML section.
                Required key: ``force_g``.
                Optional key: ``inclination_angle_deg`` (default 0).
            grid_shape: Spatial dimensions ``(nx, ny, nz, ...)``.
            **kwargs: Additional arguments including ``lattice`` (for dimension info).

        Returns:
            Gravity template array, shape ``(nx, ny, nz, 1, d)``.
        """
        # Get lattice to determine dimensionality
        lattice = kwargs.get("lattice")
        if lattice is not None:
            d = lattice.d
        else:
            # Fallback: infer from grid_shape
            d = len(grid_shape)
            d = min(d, 3)  # Cap at 3D

        nx, ny, nz = grid_shape[0], grid_shape[1], grid_shape[2] if len(grid_shape) > 2 else 1  # noqa: PLR2004

        angle_rad = jnp.deg2rad(params.get("inclination_angle_deg", 0.0))
        force_x = params["force_g"] * (-jnp.sin(angle_rad))
        force_y = params["force_g"] * jnp.cos(angle_rad)

        template = jnp.zeros((nx, ny, nz, 1, d))
        template = template.at[:, :, :, 0, 0].set(force_x)
        template = template.at[:, :, :, 0, 1].set(force_y)

        # For 3D, z-component is zero
        if d == 3:  # noqa: PLR2004
            template = template.at[:, :, :, 0, 2].set(0.0)

        return template

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
