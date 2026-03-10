"""Base class for boundary condition operators.

All boundary condition implementations inherit from
:class:`BoundaryConditionBase`, which provides the common initialisation
logic (grid, lattice, validation) shared across operators.

Subclasses must override ``__call__(self, f_streamed, f_collision)`` to
apply their specific boundary condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional


import jax.numpy as jnp

if TYPE_CHECKING:
    from app_setup.simulation_setup import SimulationSetup


@dataclass
class BoundaryConditionBase:
    """Abstract base for all boundary condition operators.

    Provides:
    - Grid and lattice initialisation from a config object.
    - ``bc_config`` storage and edge/type validation.
    - Opposite-index lookup via ``opp_indices``.

    Subclasses only need to implement ``__call__``.

    Attributes:
        bc_config: Boundary condition mapping (``{edge: type, ...}``).
        lattice: :class:`Lattice` instance built from the config.
        grid: :class:`Grid` instance built from the config.
        opp_indices: Opposite-direction index array from the lattice.
        edges: Edge slices from the grid.
    """

    bc_config: Optional[Dict[str, Any]] = field(default=None, repr=False)
    lattice: Any = field(default=None, repr=False)
    grid: Any = field(default=None, repr=False)
    opp_indices: Any = field(default=None, repr=False)
    edges: Any = field(default=None, repr=False)

    # ── Factory -----------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: "SimulationSetup",
    ) -> "BoundaryConditionBase":
        """Construct from a simulation config dataclass.

        Builds :class:`Grid` and :class:`Lattice` internally and
        validates the ``bc_config`` entries.

        Args:
            config: Typed simulation configuration.

        Returns:
            An initialised boundary condition instance.
        """
        from simulation_domain.grid import Grid
        from simulation_domain.lattice import Lattice

        grid = Grid(config.grid_shape)
        lattice = Lattice(config.lattice_type)
        bc_config = config.bc_config

        instance = cls(
            bc_config=bc_config,
            lattice=lattice,
            grid=grid,
            opp_indices=lattice.opp_indices,
            edges=grid.get_edges(),
        )

        if bc_config is not None:
            instance._validate_bc_config(bc_config)

        return instance

    # ── Validation --------------------------------------------------------

    def _get_valid_edges(self) -> set:
        """Return valid edge names derived from the grid.

        Edge names come from ``self.edges`` which is populated by
        :meth:`Grid.get_edges` (e.g. ``{"top", "bottom", "left", "right"}``
        for 2-D, plus ``"front"`` / ``"back"`` for 3-D).
        """
        if self.edges is not None:
            return set(self.edges.keys())
        return set()

    @staticmethod
    def _get_valid_types() -> set:
        """Return valid BC type names from the operator registry."""
        from app_setup.registry import get_operator_names
        return get_operator_names("boundary_condition")

    def _validate_bc_config(self, bc_config: Dict[str, Any]) -> None:
        """Raise ``ValueError`` for unknown edges or BC types.

        Valid edges are derived from ``self.edges`` (the grid) and valid
        BC types are discovered from the operator registry.  Any config
        key that is not a recognised edge is silently skipped (e.g.
        ``wetting_params``, ``hysteresis_params``, ``chemical_step``).
        """
        valid_edges = self._get_valid_edges()
        valid_types = self._get_valid_types()

        for key, value in bc_config.items():
            # Keys that don't match a grid edge are auxiliary config
            if key not in valid_edges:
                continue
            if value not in valid_types:
                raise ValueError(
                    f"Invalid BC type '{value}' for edge '{key}'. "
                    f"Must be one of {sorted(valid_types)}."
                )

    # ── Interface ---------------------------------------------------------

    def __call__(
        self, f_streamed: jnp.ndarray, f_collision: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply the boundary condition.

        Args:
            f_streamed: Distribution function after streaming.
            f_collision: Distribution function after collision (pre-stream).

        Returns:
            Updated distribution function with BCs applied.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement __call__"
        )


