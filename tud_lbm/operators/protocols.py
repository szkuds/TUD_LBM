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
from typing import runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path
    import jax.numpy as jnp
    import matplotlib.axes
    import numpy as np
    from tud_lbm.config.simulation_config import SimulationConfig
    from tud_lbm.lattice.lattice import Lattice
    from tud_lbm.pipeline.state import State
    from tud_lbm.pipeline.state import WettingState


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
        force: jnp.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute density and velocity fields.

        Args:
            f: Populations, shape ``(nx, ny, nz, q, 1)``.
            lattice: :class:`~setup.lattice.Lattice`.
            force: Optional external force field.
            **kwargs: Additional keyword arguments.
            force: Optional external force field, shape ``(nx, ny, nz, 1, d)``.
                When provided, velocity is corrected by ``u ← u + force / (2ρ)``.

        Returns:
            Without *force*: ``(rho, u)`` where
                - ``rho``: shape ``(nx, ny, nz, 1, 1)``
                - ``u``: shape ``(nx, ny, nz, 1, d)``

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


class StepOperator(Protocol):
    """Step operator — executes one full LBM time step.

    The step operator orchestrates the complete LBM algorithm:
    collision, streaming, boundary conditions, and any
    physics-specific updates (e.g., hysteresis in multiphase wetting).

    Signature::

        def step(setup, state) -> state_next
    """

    def __call__(
        self,
        setup: Any,  # setup.simulation_setup.SimulationSetup (Any avoids circular import)
        state: State,
    ) -> State:
        """Execute one LBM time step.

        Args:
            setup: :class:`~setup.simulation_setup.SimulationSetup` containing
                all pre-built operators and parameters.
            state: Current :class:`~state.state.State`.

        Returns:
            Updated :class:`~state.state.State` after one time step.
        """
        ...


class HysteresisOperator(Protocol):
    """Hysteresis operator — updates wetting state via contact angle hysteresis.

    The hysteresis operator applies dynamic contact angle adjustment based on
    velocity direction (advancing/receding). It is only called when both wetting
    and hysteresis configurations are present.

    Signature::

        def update_wetting_state(wetting, rho, setup, f_t, **kwargs) -> wetting_next
    """

    def __call__(
        self,
        wetting: Any,  # state.state.WettingState (Any avoids circular import)
        rho: jnp.ndarray,
        setup: Any,  # setup.simulation_setup.SimulationSetup (Any avoids circular import)
        f_t: jnp.ndarray,
        **kwargs: Any,
    ) -> WettingState:  # state.state.WettingState
        """Update wetting state with hysteresis.

        Args:
            wetting: Current :class:`~state.state.WettingState`.
            rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
            setup: :class:`~setup.simulation_setup.SimulationSetup`.
            f_t: Pre-step populations, shape ``(nx, ny, nz, q, 1)``.
            **kwargs: Operator-specific parameters (e.g., ``force_ext``).

        Returns:
            Updated :class:`~state.state.WettingState`.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Supporting Operators
# ══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class ForceOperator(Protocol):
    """Unified protocol for force operator modules.

    Every force module exposes setup-time ``build`` and step-time
    ``compute`` methods.
    """

    def build(self, params: Any, grid_shape: tuple[int, ...]) -> Any:
        """Construct precomputed data for the force module."""
        ...

    def compute(self, state: Any, precomputed: Any, *, diff_ops: Any = None) -> jnp.ndarray:
        """Compute the force contribution for the current state."""
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


class MultiphaseStepOperator(Protocol):
    """Bound multiphase trial-step — advances ``f_t`` by one step.

    This is the setup-bound closure stored on ``SimulationSetup.multiphase_step``.
    The setup is already captured; callers pass the current populations and
    optional physics fields.

    Signature::

        def multiphase_step(f_t, *, force_ext=None, wetting=None, ...) -> f_out
    """

    def __call__(
        self,
        f_t: jnp.ndarray,
        *,
        force_ext: jnp.ndarray | None = None,
        wetting: Any = None,
        gradient_density: Any = None,
        laplacian_density: Any = None,
    ) -> jnp.ndarray:
        """Run one multiphase trial step.

        Args:
            f_t: Pre-step populations, shape ``(nx, ny, nz, q, 1)``.
            force_ext: Optional external force, shape ``(nx, ny, nz, 1, d)``.
            wetting: Optional :class:`~state.state.WettingState`.
            gradient_density: Optional pre-built density gradient operator.
            laplacian_density: Optional pre-built density Laplacian operator.

        Returns:
            Post-BC populations, shape ``(nx, ny, nz, q, 1)``.
        """
        ...


@runtime_checkable
class ExtraState(Protocol):
    """Marker protocol for JAX-pytree-compatible extra state containers.

    Implementations are intentionally unconstrained to support both
    parameter-style containers (e.g. wetting scalars) and distribution-style
    containers (e.g. electric potential populations).
    """


@runtime_checkable
class ExtraStatePlugin(Protocol):
    """Plugin contract for initialising and updating extra ``State`` fields."""

    name: str

    def is_active(self, config: Any) -> bool:
        """Return whether this plugin should be enabled for the given config."""
        ...

    def init_state(self, setup: Any) -> dict[str, Any]:
        """Create initial extra fields merged into :class:`state.state.State`."""
        ...

    def update_state(self, setup: Any, prev_state: Any, new_state: Any, **context: Any) -> Any:
        """Apply per-step extra-state updates and return the updated state."""
        ...


class DifferentialOperator(Protocol):
    """Differential operator — computes spatial derivatives.

    Gradients and Laplacians on lattice grids, used for
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


@runtime_checkable
class EOSFunction(Protocol):
    """Bound EOS callable — evaluates bulk chemical potential for a density field.

    An ``EOSFunction`` is the *output* of an :class:`EOSOperator` builder.
    All EOS parameters are already captured in the closure; the only
    runtime argument is the density field.

    Signature::

        def eos_fn(rho) -> mu_0
    """

    def __call__(self, rho: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the bulk chemical potential μ₀(ρ).

        Args:
            rho: Density field, shape ``(nx, ny, nz, 1, 1)``.

        Returns:
            Bulk chemical potential ``μ₀``, same shape as *rho*.
        """
        ...


@runtime_checkable
class EOSOperator(Protocol):
    """EOS operator — builds a parameter-bound :class:`EOSFunction` from ``mp``.

    Each registered EOS model exposes a builder that accepts a
    :class:`~tud_lbm.operators.macroscopic.MultiphaseParams` and returns
    a closure over the relevant scalars.  The returned callable is then
    passed into the shared multiphase macroscopic computation.

    Signature::

        def build_eos(mp) -> EOSFunction
    """

    def __call__(self, mp: Any) -> EOSFunction:
        """Build a bound EOS callable for the given multiphase parameters.

        Args:
            mp: :class:`~tud_lbm.operators.macroscopic.MultiphaseParams`
                carrying all EOS-specific scalars (e.g. ``kappa``,
                ``rho_l``, ``rho_v``, ``a_eos``, …).

        Returns:
            A bound :class:`EOSFunction` ``eos_fn(rho) -> mu_0``.
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
    enabling flexible visualization strategies for different fields
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
    "EOSFunction",
    "EOSOperator",
    "EquilibriumOperator",
    "ExtraState",
    "ExtraStatePlugin",
    "ForceOperator",
    "HysteresisOperator",
    "InitialPopulationOperator",
    "InitialiserOperator",
    "MacroscopicOperator",
    "MultiphaseStepOperator",
    "PlotOperator",
    "SimulationRepository",
    "StepOperator",
    "StreamingOperator",
]
