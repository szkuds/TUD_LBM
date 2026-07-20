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
from typing import NamedTuple
import jax.numpy as jnp
from src.operators.force._gravity import _build_gravity_template
from src.registry import force_model

if TYPE_CHECKING:
    from src.pipeline.state import State


class GravityPrecomputed(NamedTuple):
    """Container for gravity precomputed data.

    Attributes:
        template: constant force field, shape (nx, ny, nz, 1, d)
        rho_l: liquid-phase reference density or None
        rho_v: vapor-phase reference density or None
    """

    template: jnp.ndarray
    rho_l: float | None
    rho_v: float | None


# ══════════════════════════════════════════════════════════════════════
# ForceOperator protocol — registry-backed module
# ══════════════════════════════════════════════════════════════════════


@force_model(name="gravity_masked_force")
class GravityForceModule:
    """Gravity force conforming to :class:`ForceOperator` protocol.

    Stateless — it uses the default no force state hooks.
    """

    @staticmethod
    def build(
        params: dict,
        grid_shape: tuple[int, ...],
        **kwargs: object,
    ) -> GravityPrecomputed:
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
        template = _build_gravity_template(params, grid_shape, **kwargs)

        # Extract optional reference densities from config (if provided).
        # getattr is intentional here: config is typed as object from **kwargs.
        config = kwargs.get("config")
        _rho_l_raw = getattr(config, "rho_l", None) if config is not None else None
        _rho_v_raw = getattr(config, "rho_v", None) if config is not None else None
        rho_l = float(_rho_l_raw) if _rho_l_raw is not None else None
        rho_v = float(_rho_v_raw) if _rho_v_raw is not None else None

        return GravityPrecomputed(template=template, rho_l=rho_l, rho_v=rho_v)

    @staticmethod
    def compute(
        state: State,
        precomputed: GravityPrecomputed,
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

        # If both liquid and vapor reference densities are available, compute
        # a smooth mask between phases and apply it to the gravity force.
        if precomputed.rho_l is not None and precomputed.rho_v is not None:
            mask = jnp.clip(
                (rho - precomputed.rho_v) / (precomputed.rho_l - precomputed.rho_v),
                0.0,
                1.0,
            )
            return -precomputed.template * rho * mask

        # Fallback: single-phase (no mask)
        return -precomputed.template * rho
