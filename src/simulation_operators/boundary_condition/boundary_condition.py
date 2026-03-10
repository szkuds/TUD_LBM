"""Composite boundary condition operator.

The ``BoundaryCondition`` class (registered as ``"standard"``) is a thin
dispatcher that applies the correct per-edge BC operator in sequence.
It inspects ``bc_config`` at construction time, resolves the matching
single-type operators from the **operator registry**, and chains their
``__call__`` methods.  No hardcoded type map is needed.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Dict, Tuple

import jax.numpy as jnp
from jax import jit

from app_setup.registry import register_operator, get_operators
from .base import BoundaryConditionBase

if TYPE_CHECKING:
    from app_setup.simulation_setup import SimulationSetup


@register_operator("boundary_condition")
class BoundaryCondition(BoundaryConditionBase):
    """Composite boundary condition — dispatches per-edge to specialised operators.

    Usage::

        bc = BoundaryCondition.from_config(config)
        # or (legacy):
        bc = BoundaryCondition(config=config)

    At construction the ``bc_config`` is inspected and one operator
    instance is created for each *distinct* BC type present.  At
    runtime ``__call__`` simply chains them.
    """

    name = "standard"

    def __init__(self, config: "SimulationSetup | None" = None, **kwargs) -> None:
        """
        Initialize the composite BoundaryCondition operator.

        Args:
            config: Configuration object containing all simulation parameters.
            **kwargs: Forwarded to :class:`BoundaryConditionBase` when
                      *config* is ``None`` (i.e. constructed via ``from_config``).
        """
        if config is not None:
            from simulation_domain.grid import Grid
            from simulation_domain.lattice import Lattice

            grid = Grid(config.grid_shape)
            lattice = Lattice(config.lattice_type)
            bc_config = config.bc_config

            super().__init__(
                bc_config=bc_config,
                lattice=lattice,
                grid=grid,
                opp_indices=lattice.opp_indices,
                edges=grid.get_edges(),
            )
            self._validate_bc_config(bc_config)
        else:
            super().__init__(**kwargs)

        # Build per-type sub-operators
        self._sub_operators: Tuple[BoundaryConditionBase, ...] = self._build_sub_operators()

    # ------------------------------------------------------------------
    # Internal: group edges by BC type and resolve from the registry
    # ------------------------------------------------------------------

    def _build_sub_operators(self) -> Tuple[BoundaryConditionBase, ...]:
        """Create one sub-operator per distinct BC type in ``bc_config``.

        BC types are resolved dynamically from the operator registry —
        no hardcoded mapping is required.
        """
        if not self.bc_config:
            return ()

        valid_edges = self._get_valid_edges()

        # Collect auxiliary (non-edge) keys from the config
        auxiliary = {
            k: v for k, v in self.bc_config.items()
            if k not in valid_edges
        }

        # Group edge entries by their BC type
        type_groups: Dict[str, Dict[str, str]] = {}
        for key, bc_type in self.bc_config.items():
            if key not in valid_edges:
                continue
            type_groups.setdefault(bc_type, {})[key] = bc_type

        # Resolve each type from the registry and instantiate
        bc_ops = get_operators("boundary_condition")
        operators: list[BoundaryConditionBase] = []

        for bc_type, edge_subset in type_groups.items():
            entry = bc_ops.get(bc_type)
            if entry is None:
                continue
            # Merge auxiliary keys (wetting_params, etc.) into the
            # per-type sub-config
            filtered_config = {**auxiliary, **edge_subset}

            op = entry.cls(
                bc_config=filtered_config,
                lattice=self.lattice,
                grid=self.grid,
                opp_indices=self.opp_indices,
                edges=self.edges,
            )
            operators.append(op)

        return tuple(operators)

    # ------------------------------------------------------------------
    # __call__ — chain sub-operators
    # ------------------------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def __call__(
        self, f_streamed: jnp.ndarray, f_collision: jnp.ndarray
    ) -> jnp.ndarray:
        for op in self._sub_operators:
            f_streamed = op(f_streamed, f_collision)
        return f_streamed
