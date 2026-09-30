"""Gravity force module.

Provides a constant-body-force implementation with no auxiliary state.

With the optional ``reference_density`` ``rho_0`` the force is only the excess
``-(rho - rho_0) g`` (Zhang, Guo & Wang 2022, Eq. 26): the ``rho_0 g`` part is
carried by the hydrostatic reference pressure ``p_g = rho_0 g.x`` in the
equilibrium and source term instead (``SimulationConfig.reference_pressure``),
so a bulk phase at ``rho_0`` feels no body force at all. The total
momentum source is unchanged, ``-grad p_g - (rho - rho_0) g = -rho g``.

Usage::

    from src.operators.force import build_forces

    forces = build_forces(config, config.grid_shape, lattice)  # [gravity_force] validated by the config
    force = forces[0].compute(state)
"""

from __future__ import annotations
import math
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


def gravity_vector(params: dict[str, Any]) -> tuple[float, float]:
    """``(g_x, g_y)`` such that the gravitational acceleration is ``-(g_x, g_y)``.

    The single statement of the inclination convention, shared by the force
    template and the reference pressure.
    """
    angle_rad = math.radians(params["inclination_angle_deg"])
    return -params["force_g"] * math.sin(angle_rad), params["force_g"] * math.cos(angle_rad)


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

    force_x, force_y = gravity_vector(params)

    template = jnp.zeros((nx, ny, nz, 1, d))
    template = template.at[:, :, :, 0, 0].set(force_x)
    return template.at[:, :, :, 0, 1].set(force_y)


# ══════════════════════════════════════════════════════════════════════
# ForceOperator protocol — registry-backed module
# ══════════════════════════════════════════════════════════════════════


@force_model(
    name="gravity_force",
    required=("force_g",),
    defaults={"inclination_angle_deg": 0.0},
    optional=("reference_density",),
    positive=("reference_density",),
)
class GravityForceModule(NamedTuple):
    """Constant gravity force, bound to its template (:class:`ForceOperator`).

    Stateless — it uses the default no force state hooks.

    Attributes:
        template: Constant force per unit density, shape ``(nx, ny, nz, 1, d)``.
        reference_density: ``rho_0`` whose weight the reference pressure
            carries; ``0.0`` without one, which is the plain ``-rho g``.
    """

    template: jnp.ndarray
    reference_density: float = 0.0

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
            params: The validated ``[gravity_force]`` section: ``force_g``,
                ``inclination_angle_deg`` (defaulted by the config) and the
                optional ``reference_density``.
            grid_shape: Spatial dimensions ``(nx, ny, nz, ...)``.
            config: Full simulation configuration (unused).
            lattice: The simulation lattice (for the velocity dimension).

        Returns:
            The force bound to its template.
        """
        return cls(
            template=_build_gravity_template(params, grid_shape, lattice),
            reference_density=float(params.get("reference_density", 0.0)),
        )

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
        return -self.template * (rho - self.reference_density)
