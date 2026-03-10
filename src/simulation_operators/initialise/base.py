"""Base class for initialisation operators.

All initialisation implementations inherit from
:class:`InitialisationBase`, which provides the common grid, lattice,
and equilibrium setup shared across operators.

Subclasses must override ``__call__`` to produce the initial
distribution function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Type, Tuple

import jax.numpy as jnp

if TYPE_CHECKING:
    from app_setup.simulation_setup import SimulationSetup


@dataclass
class InitialisationBase:
    """Abstract base for all initialisation operators.

    Provides:
    - Grid and lattice initialisation from a config object.
    - Shared equilibrium operator.
    - Registry-based type resolution.

    Subclasses only need to implement ``__call__``.

    Attributes:
        config: The simulation configuration object.
        lattice: :class:`Lattice` instance built from the config.
        grid: :class:`Grid` instance built from the config.
        equilibrium: :class:`EquilibriumWB` operator for computing f_eq.
        nx: Grid size in x-direction.
        ny: Grid size in y-direction.
        q: Number of lattice velocities.
    """

    config: Any = field(default=None, repr=False)
    lattice: Any = field(default=None, repr=False)
    grid: Any = field(default=None, repr=False)
    equilibrium: Any = field(default=None, repr=False)
    nx: int = field(default=0, repr=False)
    ny: int = field(default=0, repr=False)
    q: int = field(default=0, repr=False)

    # ── Factory -----------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: "SimulationSetup",
    ) -> "InitialisationBase":
        """Construct from a simulation config dataclass.

        Builds :class:`Grid`, :class:`Lattice`, and :class:`EquilibriumWB`
        internally.

        Args:
            config: Typed simulation configuration.

        Returns:
            An initialised instance.
        """
        from simulation_domain.grid import Grid
        from simulation_domain.lattice import Lattice
        from simulation_operators.equilibrium import EquilibriumWB

        grid = Grid(config.grid_shape)
        lattice = Lattice(config.lattice_type)
        equilibrium = EquilibriumWB(config)

        return cls(
            config=config,
            lattice=lattice,
            grid=grid,
            equilibrium=equilibrium,
            nx=grid.nx,
            ny=grid.ny,
            q=lattice.q,
        )

    # ── Registry helpers --------------------------------------------------

    @staticmethod
    def _get_valid_types() -> set:
        """Return valid initialisation type names from the operator registry."""
        from app_setup.registry import get_operator_names
        return get_operator_names("initialise")

    @staticmethod
    def resolve_type(init_type: str) -> Type["InitialisationBase"]:
        """Look up an initialisation subclass by its registered name.

        Args:
            init_type: Registered name (e.g. ``"standard"``).

        Returns:
            The registered :class:`InitialisationBase` subclass.

        Raises:
            KeyError: If ``init_type`` is not in the registry.
        """
        from app_setup.registry import get_operators
        entry = get_operators("initialise").get(init_type)
        if entry is None:
            valid = sorted(InitialisationBase._get_valid_types())
            raise KeyError(
                f"Unknown initialisation type '{init_type}'. "
                f"Must be one of {valid}."
            )
        return entry.cls

    # ── Interface ---------------------------------------------------------

    def __call__(self, **kwargs) -> jnp.ndarray:
        """Produce the initial distribution function.

        Returns:
            4-D JAX array ``f`` of shape ``(nx, ny, q, 1)`` or similar.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement __call__"
        )



