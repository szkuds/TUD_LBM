"""Gravity force module.

Provides a constant-body-force implementation with no auxiliary state.

Usage::

    from src.operators.force import build_forces

    forces = build_forces(config, config.grid_shape, lattice)  # [gravity_force] validated by the config
    force = forces[0].compute(state)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Self
import jax.numpy as jnp
from src.registry import force_model

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig
    from src.lattice.lattice import Lattice
    from src.pipeline.state import State


def _build_gravity_template(
    params: dict,
    grid_shape: tuple[int, ...],
    lattice: Lattice,
) -> jnp.ndarray:
    """Build a constant gravity template shared by gravity force variants.

    *params* has passed ``SimulationConfig`` validation against the schema
    registered with the force, so every key is present.
    """
    d = lattice.d
    nx, ny, nz = grid_shape[0], grid_shape[1], grid_shape[2] if len(grid_shape) > 2 else 1  # noqa: PLR2004

    angle_rad = jnp.deg2rad(params["inclination_angle_deg"])
    force_x = params["force_g"] * (-jnp.sin(angle_rad))
    force_y = params["force_g"] * jnp.cos(angle_rad)

    template = jnp.zeros((nx, ny, nz, 1, d))
    template = template.at[:, :, :, 0, 0].set(force_x)
    return template.at[:, :, :, 0, 1].set(force_y)


# ══════════════════════════════════════════════════════════════════════
# ForceOperator protocol — registry-backed module
# ══════════════════════════════════════════════════════════════════════


@force_model(name="gravity_force", required=("force_g",), defaults={"inclination_angle_deg": 0.0})
class GravityForceModule(NamedTuple):
    """Constant gravity force, bound to its template (:class:`ForceOperator`).

    Stateless — it uses the default no force state hooks.

    Attributes:
        template: Constant force per unit density, shape ``(nx, ny, nz, 1, d)``.
    """

    template: jnp.ndarray

    @classmethod
    def build(
        cls,
        params: dict[str, Any],
        grid_shape: tuple[int, ...],
        *,
        config: SimulationConfig,  # noqa: ARG003  # required by ForceOperator protocol
        lattice: Lattice,
    ) -> Self:
        """Build the constant gravity template.

        Args:
            params: The validated ``[gravity_force]`` section: ``force_g`` and
                ``inclination_angle_deg`` (defaulted by the config).
            grid_shape: Spatial dimensions ``(nx, ny, nz, ...)``.
            config: Full simulation configuration (unused).
            lattice: The simulation lattice (for the velocity dimension).

        Returns:
            The force bound to its template.
        """
        return cls(template=_build_gravity_template(params, grid_shape, lattice))

    def compute(self, state: State, **_kwargs: object) -> jnp.ndarray:
        """Compute gravity force (step-time, jittable).

        Args:
            state: Current simulation :class:`State`. Only ``state.f``
                is used (to compute density).
            **_kwargs: Differential operators (unused).

        Returns:
            Gravity force field, shape ``(nx, ny, nz, 1, d)``.
        """
        rho = jnp.sum(state.f, axis=-2, keepdims=True)
        return -self.template * rho
