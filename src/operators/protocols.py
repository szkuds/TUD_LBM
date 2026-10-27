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
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import Self
from typing import runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    import jax.numpy as jnp
    import matplotlib.axes
    import numpy as np
    from src.config.simulation_config import SimulationConfig
    from src.lattice.lattice import Lattice
    from src.pipeline.state import State


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
        source: jnp.ndarray | None = None,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Compute post-collision distribution.

        Args:
            f: Populations, shape ``(nx, ny, nz, q, 1)``.
            feq: Equilibrium distribution, shape ``(nx, ny, nz, q, 1)``.
            tau: Relaxation time (> 0.5).
            source: Optional forcing source term, shape ``(nx, ny, nz, q, 1)``.
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
        /,
    ) -> jnp.ndarray:
        """Propagate populations across the domain.

        Args:
            f: Populations, shape ``(nx, ny, nz, q, 1)``.
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
            rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
            u: Velocity field, shape ``(nx, ny, nz, 1, d)`` where d ∈ {2, 3}.
            lattice: :class:`~setup.lattice.Lattice` with weights ``w``
                and velocity vectors ``c``.

        Returns:
            Equilibrium distribution ``feq``, shape ``(nx, ny, nz, q, 1)``.
        """
        ...


class MacroscopicOperator(Protocol):
    """Macroscopic operator — computes ``(f, lattice) → (rho, u, force, pressure)``.

    Every macroscopic field is a function of the moments of the population
    distribution, and every macroscopic operator returns the same four:

    * ``rho`` — zeroth moment, shape ``(nx, ny, nz, 1, 1)``;
    * ``u`` — first moment over ``rho``, force-corrected when a force acts,
      shape ``(nx, ny, nz, 1, d)``;
    * ``force`` — the total force, shape ``(nx, ny, nz, 1, d)``, or ``None``
      when none acts (single-phase without an external force);
    * ``pressure`` — the bulk pressure, shape ``(nx, ny, nz, 1, 1)``. The
      multiphase operator takes it from the EOS's registered ``p_0(rho)``;
      single-phase uses the lattice ideal gas ``cs^2 * rho``. Either way a
      hydrostatic stratification shows up in it.

    Operator-specific extras (an external force, multiphase parameters, bound
    differential operators) follow the two shared positional arguments.

    Signature::

        def macroscopic_op(f, lattice, /, ...) -> (rho, u, force, pressure)
    """

    def __call__(
        self,
        f: jnp.ndarray,
        lattice: Lattice,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray]:
        """Compute the macroscopic fields from the populations.

        Args:
            f: Populations, shape ``(nx, ny, nz, q, 1)``.
            lattice: :class:`~src.lattice.lattice.Lattice`.
            *args: Operator-specific positional arguments.
            **kwargs: Operator-specific keyword arguments.

        Returns:
            ``(rho, u, force, pressure)``; see the class docstring.
        """
        ...


class BoundaryCondition(Protocol):
    """One registered boundary condition, applied to a single domain edge.

    Signature::

        def apply_bc(f_streamed, f_collision, lattice, edge) -> f_bc
    """

    def __call__(
        self,
        f_streamed: jnp.ndarray,
        f_collision: jnp.ndarray,
        lattice: Lattice,
        edge: str,
        /,
    ) -> jnp.ndarray:
        """Apply this boundary condition on *edge*.

        Args:
            f_streamed: Post-streaming populations.
            f_collision: Post-collision populations (for symmetry BC).
            lattice: :class:`~src.lattice.lattice.Lattice`.
            edge: Edge name, e.g. ``"bottom"``.

        Returns:
            Populations with the boundary condition applied on *edge*.
        """
        ...


class BoundaryOperator(Protocol):
    """Boundary-condition operator — applies every edge's BC to populations.

    Boundary conditions enforce Dirichlet/Neumann constraints or flux
    periodicity at domain edges. They are applied post-streaming.

    Signature::

        def apply_bcs(f_stream, f_col) -> f_bc
    """

    def __call__(
        self,
        f_stream: jnp.ndarray,
        f_col: jnp.ndarray,
        /,
    ) -> jnp.ndarray:
        """Apply boundary conditions to post-streaming populations.

        Args:
            f_stream: Post-streaming populations.
            f_col: Post-collision populations (for symmetry BC).

        Returns:
            Populations with boundary conditions applied.
        """
        ...


class ObstacleOperator(Protocol):
    """Obstacle operator — applies bounce-back at interior solid cells.

    Unlike :class:`BoundaryOperator`, which applies whole-edge BC rules,
    this operator reverses populations at an arbitrary per-cell mask of
    solid (obstacle) nodes. Applied post-streaming, pre-edge-BC.

    Signature::

        def apply_obstacle(f_stream, f_col) -> f_stream
    """

    def __call__(
        self,
        f_stream: jnp.ndarray,
        f_col: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply bounce-back at masked interior solid cells.

        Args:
            f_stream: Post-streaming populations.
            f_col: Post-collision populations.

        Returns:
            Populations with obstacle bounce-back applied.
        """
        ...


class InitialiserOperator(Protocol):
    """Initialiser operator — creates the initial distribution ``f``.

    Initialisation strategies include:
    - "standard": rest equilibrium ``f_eq(ρ_0, u_0)`` where ρ_0 = 1, u_0 = 0
    - "init_from_file": load from an NPZ file
    - Multiphase variants: tanh density profile

    Signature::

        def init_fn(grid_shape, lattice, **kwargs) -> f
    """

    def __call__(
        self,
        grid_shape: tuple[int, int, int],
        lattice: Lattice,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Initialise the distribution function.

        Args:
            grid_shape: Grid dimensions ``(nx, ny, nz)``.
            lattice: :class:`~setup.lattice.Lattice`.
            **kwargs: Initialiser-specific keyword arguments
                (e.g., ``density``, ``rho_l``, ``rho_v``, ``interface_width``,
                ``npz_path``).

        Returns:
            Initial population distribution, shape ``(nx, ny, nz, q, 1)``.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Supporting Operators
# ══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class ForceOperator(Protocol):
    """A force bound to the payload it was built with.

    A registered force class is its own factory: the classmethod ``build``
    reads the config-validated ``*_force`` section and returns an instance
    carrying whatever ``compute`` needs (a template, band edges, electric
    parameters). ``SimulationSetup.forces`` holds these instances, so there is
    no separate container pairing a force with its payload.
    """

    @classmethod
    def build(
        cls,
        params: dict[str, Any],
        grid_shape: tuple[int, ...],
        *,
        config: SimulationConfig,
        lattice: Lattice,
    ) -> Self:
        """Build the bound force from a config-validated parameter section (setup-time)."""
        ...

    def compute(
        self,
        state: State,
        *,
        gradient_standard: DifferentialOperator | None = None,
        gradient_density: DifferentialOperator | None = None,
        laplacian_density: DifferentialOperator | None = None,
    ) -> jnp.ndarray:
        """Return this force's hooks, shape ``(nx, ny, nz, 1, d)`` (step-time, jittable)."""
        ...


class SourceTermOperator(Protocol):
    """Couples the total force into the populations as a collision source term.

    Signature::

        def source(rho, u, force, lattice, *, gradient) -> src   # (nx, ny, nz, q, 1)
    """

    def __call__(
        self,
        rho: jnp.ndarray,
        u: jnp.ndarray,
        force: jnp.ndarray,
        lattice: Lattice,
        *,
        gradient: DifferentialOperator,
    ) -> jnp.ndarray:
        """Compute the source term."""
        ...


class InitialPopulationOperator(Protocol):
    """Bound initialiser — builds the initial ``f`` for a fixed setup.

    This is the setup-bound closure stored on ``SimulationSetup.initial_f_fn``.
    Unlike ``InitialiserOperator``, the grid shape and lattice are already
    captured; callers only supply optional overrides.

    Signature::

        def initial_f_fn(init_kwargs=None) -> f
    """

    def __call__(self, init_kwargs: dict | None = None) -> jnp.ndarray:
        """Build the initial population distribution.

        Args:
            init_kwargs: Optional keyword overrides (e.g. ``density``,
                ``rho_l``, ``npz_path``).

        Returns:
            Initial populations, shape ``(nx, ny, nz, q, 1)``.
        """
        ...


class DifferentialOperator(Protocol):
    """Differential operator bound to its run — what a consumer calls.

    The macroscopic operator, the source term and the forces receive the
    gradient / Laplacian with the addon substates already bound (the kernel
    binds ``State.ext`` with :func:`functools.partial`), so they see a plain
    field-to-field map.

    Signature::

        def diff_op(grid) -> derivative_field
    """

    def __call__(self, grid: jnp.ndarray) -> jnp.ndarray:
        """Compute a spatial derivative of *grid*.

        Args:
            grid: Scalar field, shape ``(nx, ny, nz, 1, 1)``.

        Returns:
            Derivative field, shape ``(nx, ny, nz, 1, 2)`` for a gradient or
            ``(nx, ny, nz, 1, 1)`` for a Laplacian.
        """
        ...


class ExtDifferentialOperator(Protocol):
    """Differential operator as built — a stencil closure that also reads ``State.ext``.

    Built by :func:`~src.operators.differential.build_differential`, which
    captures the lattice, the pad modes and an optional *ghost fill*: the
    composed ``ghost_fill`` slot of the active addons, applied between
    padding and the stencil. The fill reads the addon substates from *ext*
    (the wetting wall reads ``ext["wetting"]``); an operator built without a
    fill ignores *ext*.

    Signature::

        def diff_op(grid, ext) -> derivative_field
    """

    def __call__(self, grid: jnp.ndarray, ext: Mapping[str, Any]) -> jnp.ndarray:
        """Compute a spatial derivative of *grid* under the addon substates *ext*.

        Args:
            grid: Scalar field, shape ``(nx, ny, nz, 1, 1)``.
            ext: Addon substates, ``State.ext`` (or a candidate variant of it).

        Returns:
            Derivative field, shape ``(nx, ny, nz, 1, 2)`` for a gradient or
            ``(nx, ny, nz, 1, 1)`` for a Laplacian.
        """
        ...


@runtime_checkable
class EosOperator(Protocol):
    """Bound EOS callable — a bulk thermodynamic function of the density field.

    Each EOS registers two, both derivatives of the same bulk free energy and
    built from the same parameters: the chemical potential ``mu_0(rho)``
    (:func:`~src.operators.macroscopic.eos.build_eos_fn`) and the bulk pressure
    ``p_0(rho)`` (:func:`~src.operators.macroscopic.eos.build_pressure_fn`).
    All EOS parameters are captured in the closure; the only runtime argument
    is the density field.

    Signature::

        def eos_fn(rho) -> mu_0 or p_0
    """

    def __call__(self, rho: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the bound thermodynamic function of density.

        Args:
            rho: Density field, shape ``(nx, ny, nz, 1, 1)``.

        Returns:
            ``mu_0`` or ``p_0``, same shape as *rho*.
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
        field_names: tuple[str, ...] | None = None,
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


class PlotOperator(Protocol):
    """Structural contract for Matplotlib plot operators.

    Plot operators render simulation snapshots onto matplotlib axes,
    enabling flexible visualisation strategies for different fields
    and use cases.

    Signature::

        class MyPlotter(PlotOperatorProtocol):
            def __call__(self, ax, data, timestep):
                # render to axes
    """

    name: str
    config: SimulationConfig

    def __init__(
        self,
        config: SimulationConfig,
        data_dir: str | Path | None = None,
    ) -> None:
        """Initialise the plot operator.

        Args:
            config: :class:`~config.simulation_config.SimulationConfig`.
            data_dir: Optional directory path for data-dependent visualisations.
        """
        ...

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        """Return whether this operator can render the provided snapshot.

        Args:
            data: Snapshot dictionary with field names as keys (e.g., ``"rho"``, ``"u"``).

        Returns:
            ``True`` if all required fields are present; ``False`` otherwise.
        """
        ...

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        """Render one panel onto the provided axes.

        Args:
            ax: Matplotlib axes object to draw onto.
            data: Snapshot dictionary with computed fields.
            timestep: Current iteration number (for annotations).
        """
        ...


__all__ = [
    # Core operators
    "BoundaryOperator",
    "CollisionOperator",
    "ConfigReader",
    "DifferentialOperator",
    "EosOperator",
    "EquilibriumOperator",
    "ExtDifferentialOperator",
    "ForceOperator",
    "InitialPopulationOperator",
    "InitialiserOperator",
    "MacroscopicOperator",
    "PlotOperator",
    "SimulationRepository",
    "SourceTermOperator",
    "StreamingOperator",
]
