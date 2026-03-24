"""Protocol (structural) types for LBM operators.

These protocols define the contract that each operator category must fulfil.
They enable loose coupling: code depending on `CollisionOperator` can work
with any function/class implementing that protocol, without importing
the specific implementation.

Design principle: Operator protocols are intentionally minimal — they
capture the bare essentials (signatures, docstrings) without dictating
implementation details like decorators or registry membership.

Usage::

    from operators.protocols import CollisionOperator
    from registry import get_operators

    collision_ops = get_operators("collision_models")
    bgk_fn = collision_ops["bgk"].target

    # Static type-checkers and isinstance() will accept bgk_fn
    # as a CollisionOperator
    def my_collision_logic(collision_op: CollisionOperator):
        ...
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import jax.numpy as jnp

if False:  # TYPE_CHECKING
    from setup.lattice import Lattice
    from setup.simulation_setup import BCMasks, SimulationSetup
    from state.state import State, WettingState
    from operators.differential.operators import DifferentialOperators


# ══════════════════════════════════════════════════════════════════════════════
# Core LBM Operators
# ══════════════════════════════════════════════════════════════════════════════


class CollisionOperator(Protocol):
    """Collision operator — transforms ``(f, feq, tau) → f_col``.

    The collision step replaces non-conserved moments with their
    equilibrium values, relaxed toward equilibrium with time scale ``tau``.

    Signature::

        def collide(f, feq, tau, source=None, ...) -> f_col
    """

    def __call__(
        self,
        f: jnp.ndarray,
        feq: jnp.ndarray,
        tau: float,
        source: Optional[jnp.ndarray] = None,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Compute post-collision distribution.

        Args:
            f: Populations, shape ``(nx, ny, q, 1)``.
            feq: Equilibrium distribution, shape ``(nx, ny, q, 1)``.
            tau: Relaxation time (> 0.5).
            source: Optional forcing source term, shape ``(nx, ny, q, 1)``.
            **kwargs: Operator-specific parameters.

        Returns:
            Post-collision populations, same shape as *f*.
        """
        ...


class StreamingOperator(Protocol):
    """Streaming operator — propagates populations along velocity directions.

    The streaming step shifts each population component ``f_i`` along the
    direction of its lattice velocity ``c_i``, using periodic boundary
    conditions across the domain (boundary conditions are applied afterward).

    Signature::

        def stream(f, lattice) -> f_streamed
    """

    def __call__(
        self,
        f: jnp.ndarray,
        lattice: Lattice,
    ) -> jnp.ndarray:
        """Propagate populations across the domain.

        Args:
            f: Populations, shape ``(nx, ny, q, 1)``.
            lattice: :class:`~setup.lattice.Lattice` with velocity vectors ``c``.

        Returns:
            Post-streaming populations, same shape as *f*.
        """
        ...


class EquilibriumOperator(Protocol):
    """Equilibrium operator — computes ``(rho, u, lattice) → feq``.

    The equilibrium distribution is the rest state toward which the
    collision operator relaxes the system. It encodes the
    hydrodynamic moment structure.

    Signature::

        def compute_equilibrium(rho, u, lattice) -> feq
    """

    def __call__(
        self,
        rho: jnp.ndarray,
        u: jnp.ndarray,
        lattice: Lattice,
    ) -> jnp.ndarray:
        """Compute the equilibrium distribution.

        Args:
            rho: Density field, shape ``(nx, ny, 1, 1)``.
            u: Velocity field, shape ``(nx, ny, 1, d)`` where d ∈ {2, 3}.
            lattice: :class:`~setup.lattice.Lattice` with weights ``w``
                and velocity vectors ``c``.

        Returns:
            Equilibrium distribution ``feq``, shape ``(nx, ny, q, 1)``.
        """
        ...


class MacroscopicOperator(Protocol):
    """Macroscopic operator — computes ``(f, lattice) → (rho, u, ...)``.

    Macroscopic fields are the moments of the population distribution,
    computed via summation over velocity directions.

    Signature::

        def compute_macroscopic(f, lattice, force=None) -> (rho, u) or (rho, u, force)
    """

    def __call__(
        self,
        f: jnp.ndarray,
        lattice: Lattice,
        force: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute density and velocity fields.

        Args:
            f: Populations, shape ``(nx, ny, q, 1)``.
            lattice: :class:`~setup.lattice.Lattice`.
            force: Optional external force field, shape ``(nx, ny, 1, d)``.
                When provided, velocity is corrected by ``u ← u + force / (2ρ)``.

        Returns:
            Without *force*: ``(rho, u)`` where
                - ``rho``: shape ``(nx, ny, 1, 1)``
                - ``u``: shape ``(nx, ny, 1, d)``

            With *force*: ``(rho, u_eq, force)`` where *u_eq* includes the force correction.
        """
        ...


class BoundaryOperator(Protocol):
    """Boundary-condition operator — applies edge BC rules to populations.

    Boundary conditions enforce Dirichlet/Neumann constraints or flux
    periodicity at domain edges. They are applied post-streaming.

    Signature::

        def apply_bc(f_stream, f_col, bc_masks) -> f_bc
    """

    def __call__(
        self,
        f_stream: jnp.ndarray,
        f_col: jnp.ndarray,
        bc_masks: Any,  # BCMasks NamedTuple
    ) -> jnp.ndarray:
        """Apply boundary conditions to post-streaming populations.

        Args:
            f_stream: Post-streaming populations.
            f_col: Post-collision populations (for symmetry BC).
            bc_masks: Pre-computed edge masks from
                :class:`~setup.simulation_setup.BCMasks`.

        Returns:
            Populations with boundary conditions applied.
        """
        ...


class InitialiserOperator(Protocol):
    """Initialiser operator — creates the initial distribution ``f``.

    Initialisation strategies include:
    - "standard": rest equilibrium ``f_eq(ρ_0, u_0)`` where ρ_0 = 1, u_0 = 0
    - "init_from_file": load from an NPZ file
    - Multiphase variants: tanh density profile

    Signature::

        def init_fn(nx, ny, lattice, **kwargs) -> f
    """

    def __call__(
        self,
        nx: int,
        ny: int,
        lattice: Lattice,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Initialise the distribution function.

        Args:
            nx: Grid width.
            ny: Grid height.
            lattice: :class:`~setup.lattice.Lattice`.
            **kwargs: Initialiser-specific keyword arguments
                (e.g., ``density``, ``rho_l``, ``rho_v``, ``interface_width``,
                ``npz_path``).

        Returns:
            Initial population distribution, shape ``(nx, ny, q, 1)``.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Supporting Operators
# ══════════════════════════════════════════════════════════════════════════════


class ForcingOperator(Protocol):
    """Generic forcing operator — computes external forces.

    Examples: gravity (body force), electric force (charged fluids).

    Signature::

        def compute_force(state_fields, ...) → force_field
    """

    def __call__(self, **kwargs: Any) -> jnp.ndarray:
        """Compute external force field.

        Args:
            **kwargs: Force-specific parameters (e.g., ``rho``, ``h``, ``template``).

        Returns:
            Force field, typically shape ``(nx, ny, 1, d)``.
        """
        ...


class DifferentialOperator(Protocol):
    """Differential operator — computes spatial derivatives.

    Typically gradients and Laplacians on lattice grids, used for
    multiphase chemical potential and interfacial stress.

    Signature::

        def compute_derivative(field) → derivative_field
    """

    def __call__(self, field: jnp.ndarray) -> jnp.ndarray:
        """Compute a spatial derivative.

        Args:
            field: Scalar or vector field, shape ``(nx, ny, 1, 1)`` or ``(nx, ny, 1, 2)``.

        Returns:
            Derivative field, matching or broadened shape.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# IO / Persistence Ports
# ══════════════════════════════════════════════════════════════════════════════


class SimulationRepository(Protocol):
    """Persistence port: writes simulation state and metadata to disk.

    Abstracts the storage mechanism (HDF5, NumPy .npz, Parquet, etc.).

    Typical operations:
    - Save trajectory snapshots at specified intervals
    - Write metadata (config, simulation parameters)
    - Recover state for restart

    Signature::

        class MyRepository(SimulationRepository):
            def save_snapshot(self, state, time_step, field_names):
                # write to disk
            def load_snapshot(self, time_step):
                # read from disk and return State object
    """

    def save_snapshot(
        self,
        state: State,
        time_step: int,
        field_names: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """Persist a simulation state snapshot.

        Args:
            state: Current :class:`~state.state.State`.
            time_step: Current iteration number (for naming/indexing).
            field_names: Which fields to save (e.g., ``("rho", "u")``).
                ``None`` means save all fields.
        """
        ...

    def load_snapshot(self, time_step: int) -> State:
        """Load a previously saved snapshot.

        Args:
            time_step: Iteration number of the snapshot to retrieve.

        Returns:
            Reconstructed :class:`~state.state.State`.
        """
        ...


class ConfigReader(Protocol):
    """Parsing port: reads and validates configuration from external format.

    Abstracts the input format (TOML, JSON, YAML, dict, etc.).

    Typical operations:
    - Parse a config file
    - Validate against schema
    - Return a :class:`~config.simulation_config.SimulationConfig`

    Signature::

        class TomlConfigReader(ConfigReader):
            def load(self, path):
                # read TOML file and return SimulationConfig
    """

    def load(self, source: str) -> Any:  # SimulationConfig
        """Read and parse a configuration.

        Args:
            source: Configuration source (filepath, dict, URL, etc.).

        Returns:
            A validated :class:`~config.simulation_config.SimulationConfig`.
        """
        ...


__all__ = [
    # Core operators
    "CollisionOperator",
    "StreamingOperator",
    "EquilibriumOperator",
    "MacroscopicOperator",
    "BoundaryOperator",
    "InitialiserOperator",
    "ForcingOperator",
    "DifferentialOperator",
    # Ports
    "SimulationRepository",
    "ConfigReader",
]
