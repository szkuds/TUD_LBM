"""Buoyancy-referenced gravity with a hydrostatic reference pressure (Zhang, Guo & Wang 2022).

``[gravity_referenced_force]``: the force is only the excess ``-(rho - rho_0) g``
over a reference density ``rho_0`` (Eq. 26). The ``rho_0 g`` part is carried
instead by the reference pressure ``p_g = rho_0 g.x`` in the second moment of the
equilibrium and as ``-grad p_g`` in the source term's velocity-force product, which
this force's payload supplies to ``build_improved_equilibrium``
(``equilibrium/_equilibrium_improved_well_balanced.py``) and ``build_referenced_source``
(``source_term/_source_well_balanced_referenced.py``). The total momentum
source is unchanged, ``-grad p_g - (rho - rho_0) g = -rho g``, so a bulk phase at
``rho_0`` feels no body force.

The constant template is the plain gravity force's (:mod:`._gravity`, imported,
never edited), so both forces share one inclination convention. ``p_g`` is measured
from the domain centre to keep ``|p_g|`` small; an additive constant has no
dynamical effect.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Self
import jax.numpy as jnp
import numpy as np
from src.operators.force._gravity import _build_gravity_template
from src.registry import force_model

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig
    from src.lattice.lattice import Lattice
    from src.pipeline.state import State


@force_model(
    name="gravity_referenced_force",
    required=("force_g", "reference_density"),
    defaults={"inclination_angle_deg": 0.0},
    positive=("reference_density",),
)
class GravityReferencedForceModule(NamedTuple):
    """Excess gravity over ``rho_0``, bound to its template and reference pressure.

    Attributes:
        template: Constant force per unit density ``g``, shape ``(nx, ny, nz, 1, d)``;
            the acceleration is ``-template``.
        reference_density: ``rho_0``.
        pressure: ``p_g = rho_0 g.x``, shape ``(nx, ny, nz, 1, 1)``.
        pressure_gradient: ``grad p_g = rho_0 g``, shape ``(nx, ny, nz, 1, d)``.
    """

    template: jnp.ndarray
    reference_density: float
    pressure: jnp.ndarray
    pressure_gradient: jnp.ndarray

    @classmethod
    def build(
        cls,
        params: dict[str, Any],
        grid_shape: tuple[int, ...],
        *,
        config: SimulationConfig,  # noqa: ARG003  # required by ForceOperator protocol
        lattice: Lattice,
    ) -> Self:
        """Build the template and the reference pressure from the validated section.

        Args:
            params: The validated ``[gravity_referenced_force]`` section: ``force_g``,
                ``reference_density`` and ``inclination_angle_deg`` (defaulted).
            grid_shape: Spatial dimensions ``(nx, ny, nz, ...)``.
            config: Full simulation configuration (unused).
            lattice: The simulation lattice (for the velocity dimension).

        Returns:
            The force bound to its payload.
        """
        template = _build_gravity_template(params, grid_shape, lattice)
        rho_0 = float(params["reference_density"])
        nx, ny = grid_shape[0], grid_shape[1]
        x, y = np.meshgrid(np.arange(nx) - (nx - 1) / 2.0, np.arange(ny) - (ny - 1) / 2.0, indexing="ij")
        coordinates = jnp.asarray(np.stack([x, y], axis=-1))[:, :, None, None, :]  # (nx, ny, 1, 1, 2)
        pressure_gradient = rho_0 * template
        return cls(
            template=template,
            reference_density=rho_0,
            pressure=jnp.sum(pressure_gradient * coordinates, axis=-1, keepdims=True),
            pressure_gradient=pressure_gradient,
        )

    def compute(self, state: State, **_kwargs: object) -> jnp.ndarray:
        """Compute the excess gravity ``-(rho - rho_0) g`` (step-time, jittable).

        Args:
            state: Current simulation :class:`State`; only ``state.f`` is used.
            **_kwargs: Differential operators (unused).

        Returns:
            Force field, shape ``(nx, ny, nz, 1, d)``.
        """
        rho = jnp.sum(state.f, axis=-2, keepdims=True)
        return -self.template * (rho - self.reference_density)
