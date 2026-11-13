"""Every interface in the codebase: operator protocols, addon hooks and the base classes.

One file holds the contracts so a reader finds every extension point in one
place:

* **Operator protocols** — the structural type each registry kind's
  implementations satisfy (``CollisionOperator``, ``BoundaryCondition``, ...).
* **Addon hooks** — the slot types of :class:`~src.addons.hooks.Hooks`.
* **Analysis** — :data:`DimensionlessNumberOperator`.
* **Wetting** — :class:`OptaxLike`, the part of an ``optax`` optimiser the
  hysteresis loop uses.
* **Base classes** — the ABCs subclassed by plotting operators, config
  adapters and output writers, shared implementation included.

Import constraint: at runtime this module imports only the standard library,
``jax`` and ``numpy``. Everything else (matplotlib, ``SimulationConfig``,
``State``, ...) is imported under ``TYPE_CHECKING`` or inside a method, so
importing the core never pulls in plotting and no import cycle forms here.

Usage::

    from src.protocols import CollisionOperator
    from src.registry import get_operators

    def build_collision_fn(scheme: str) -> CollisionOperator:
        return get_operators("collision_models")[scheme].target
"""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Protocol
from typing import Self
from typing import TypeVar
from typing import runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    import jax.numpy as jnp
    import matplotlib.axes
    import matplotlib.figure
    import numpy as np
    from src.config.boundary import Edge
    from src.config.simulation_config import SimulationConfig
    from src.lattice.lattice import Lattice
    from src.operators.wetting._params import WettingParams
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state import State
    from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs

    #: An obstacle shape: ``(params, grid_shape) -> mask``, a boolean ``(nx, ny, nz, 1, 1)``
    #: solid-cell array, built once at setup time in NumPy.
    ObstacleMaskBuilder = Callable[[dict[str, Any], tuple[int, int, int]], np.ndarray]


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
    """Equilibrium operator — computes ``(rho, u, lattice[, pressure]) → feq``.

    The equilibrium distribution is the rest state toward which the
    collision operator relaxes the system. It encodes the
    hydrodynamic moment structure.

    *pressure* is the isotropic part of the second moment when the run supplies
    one (the macroscopic operator's ``lattice_pressure``). An equilibrium whose
    pressure is fixed by its own form takes it and ignores it.

    Signature::

        def compute_equilibrium(rho, u, lattice, pressure=None) -> feq
    """

    def __call__(
        self,
        rho: jnp.ndarray,
        u: jnp.ndarray,
        lattice: Lattice,
        pressure: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute the equilibrium distribution.

        Args:
            rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
            u: Velocity field, shape ``(nx, ny, nz, 1, d)`` where d ∈ {2, 3}.
            lattice: :class:`~setup.lattice.Lattice` with weights ``w``
                and velocity vectors ``c``.
            pressure: Pressure for the second moment, shape ``(nx, ny, nz, 1, 1)``,
                or ``None`` when the run supplies none.

        Returns:
            Equilibrium distribution ``feq``, shape ``(nx, ny, nz, q, 1)``.
        """
        ...


class MacroscopicOperator(Protocol):
    """Macroscopic operator — computes ``(f, lattice) → (rho, u, force, pressure, lattice_pressure)``.

    Every macroscopic field is a function of the moments of the population
    distribution, and every macroscopic operator returns the same five:

    * ``rho`` — zeroth moment, shape ``(nx, ny, nz, 1, 1)``;
    * ``u`` — first moment over ``rho``, force-corrected when a force acts,
      shape ``(nx, ny, nz, 1, d)``;
    * ``force`` — the total force, shape ``(nx, ny, nz, 1, d)``, or ``None``
      when none acts (single-phase without an external force);
    * ``pressure`` — the bulk pressure, shape ``(nx, ny, nz, 1, 1)``. The
      multiphase operator takes it from the EOS's registered ``p_0(rho)``;
      single-phase uses the lattice ideal gas ``cs^2 * rho``. Either way a
      hydrostatic stratification shows up in it;
    * ``lattice_pressure`` — the pressure the step hands to the equilibrium,
      shape ``(nx, ny, nz, 1, 1)``, or ``None`` when the equilibrium needs none
      supplied. It is not the physical pressure above: a multiphase run with a
      reference density returns ``rho_0 * (mu - mu_b)`` here.

    Both operators share one signature. The differential operators come with
    the addon substates already bound; a single-phase operator ignores them,
    and a multiphase one has its parameters bound at build time.

    Signature::

        def macroscopic_op(f, lattice, force, *, gradient_standard, laplacian_density)
            -> (rho, u, force, pressure, lattice_pressure)
    """

    def __call__(
        self,
        f: jnp.ndarray,
        lattice: Lattice,
        force: jnp.ndarray | None,
        *,
        gradient_standard: DifferentialOperator,
        laplacian_density: DifferentialOperator,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray, jnp.ndarray | None]:
        """Compute the macroscopic fields from the populations.

        Args:
            f: Populations, shape ``(nx, ny, nz, q, 1)``.
            lattice: :class:`~src.lattice.lattice.Lattice`.
            force: External force, shape ``(nx, ny, nz, 1, d)``, or ``None``.
            gradient_standard: Gradient for the chemical potential, ``grad(mu)``.
            laplacian_density: Laplacian of the density, ``lap(rho)``.

        Returns:
            ``(rho, u, force, pressure, lattice_pressure)``; see the class docstring.
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
        edge: Edge,
        /,
    ) -> jnp.ndarray:
        """Apply this boundary condition on *edge*.

        Args:
            f_streamed: Post-streaming populations.
            f_collision: Post-collision populations (for symmetry BC).
            lattice: :class:`~src.lattice.lattice.Lattice`.
            edge: The domain face.

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
# Addon hooks
# ══════════════════════════════════════════════════════════════════════════════

#: An addon's substate, ``State.ext[name]``.
S = TypeVar("S")

if TYPE_CHECKING:
    #: Rewrites the ghost layer of a padded scalar field from the addon's own substate.
    GhostFill = Callable[[jnp.ndarray, S], jnp.ndarray]
    #: The addons' fills composed: reads each substate off ``ext`` by name.
    ExtGhostFill = Callable[[jnp.ndarray, Mapping[str, Any]], jnp.ndarray]
    #: Runs after the kernel: ``(setup, state before the step, state after it) -> state``.
    PostStepHook = Callable[[SimulationSetup, State, State], State]
    #: Builds the addon's initial substate, ``State.ext[name]``.
    InitStateHook = Callable[[SimulationSetup], S]


# ══════════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════════

if TYPE_CHECKING:
    #: A registered ``dimensionless`` operator: pure, total, and ``None`` rather
    #: than raising when this run cannot supply the number.
    DimensionlessNumberOperator = Callable[[DimensionlessInputs], float | None]


# ══════════════════════════════════════════════════════════════════════════════
# Wetting
# ══════════════════════════════════════════════════════════════════════════════


class OptaxLike(Protocol):
    """Minimal structural type for optax-compatible optimisers."""

    def init(self, params: WettingParams) -> object:
        """Initial optimiser state for *params*."""
        ...

    def update(
        self, updates: WettingParams, state: object, params: WettingParams | None = ...
    ) -> tuple[WettingParams, object]:
        """Transformed updates and the next optimiser state."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Base classes
# ══════════════════════════════════════════════════════════════════════════════


class PlotOperator(ABC):
    """Base class for a single-panel plotting operator."""

    name: str

    #: Excluded from the "every registered plotting operator" default that
    #: :class:`~src.simulation_io.plotting.figure_builder.FigureBuilder` falls
    #: back on when ``plot_fields`` is unset. An opt-in operator only renders
    #: when it is named explicitly.
    opt_in: bool = False

    #: May be named in ``overlay_fields``: :meth:`draw_overlay` then draws this
    #: operator on top of every other field panel. Overlays are a capability of
    #: a plotting operator, not a separate registry kind.
    supports_overlay: bool = False

    #: Only drawn as an overlay: never a standalone panel, and hidden from field
    #: selection. Implies :attr:`supports_overlay`.
    overlay_only: bool = False

    #: Wording for the interactive overlay question; falls back to :attr:`name`.
    overlay_label: str = ""

    #: Whether overlays are drawn onto this operator's own panel. Independently
    #: of this flag, an overlay is never drawn onto its own standalone panel, so
    #: an overlay-capable panel (``interface``) still carries the *other*
    #: overlays without drawing itself twice.
    accepts_overlays: bool = True

    def __init__(self, config: SimulationConfig, data_dir: str | Path | None = None) -> None:
        """Initialise the plot operator with config and optional data directory.

        Args:
            config: Simulation configuration object.
            data_dir: Optional path to data directory for loading snapshots.
        """
        self.config = config
        self.data_dir = Path(data_dir) if data_dir is not None else None

    @abstractmethod
    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        """Draw this operator on the provided axes."""

    @classmethod
    def overlay_prompt_default(cls, config: SimulationConfig) -> bool | None:  # noqa: ARG003
        """Default answer to the interactive overlay question for *config*.

        ``None`` means the question is not offered at all, because the overlay
        has nothing to show for this run. Only consulted when
        :attr:`supports_overlay` is set.
        """
        return True

    def draw_overlay(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        """Draw this operator on top of another panel's axes.

        Only called when :attr:`supports_overlay` is set. Must add artists only —
        no colorbar, title or axis-limit change — so the host panel is unchanged.
        """
        msg = f"Plot operator {self.name!r} does not support being drawn as an overlay."
        raise NotImplementedError(msg)

    def is_available(self, data: dict[str, np.ndarray]) -> bool:  # noqa: ARG002
        """Whether this operator has enough data to render."""
        return True


class AnalysisPlot(ABC):
    """Base class for analysis plots computed from saved snapshot history."""

    name: str
    is_multi_panel: bool = False

    def __init__(self, config: SimulationConfig | None = None) -> None:
        """Initialise with optional simulation config.

        Args:
            config: Simulation configuration object. Required by config-aware
                operators (for example, capillary-number or normalised
                contact-line operators). Operators that do not need config can
                ignore it.
        """
        self.config = config
        self._primed_xlim: tuple[float, float] | None = None
        self._primed_ylims: list[tuple[float, float]] | None = None

    @abstractmethod
    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute time-series arrays from snapshot files."""

    @abstractmethod
    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Render the full analysis plot from precomputed arrays."""

    def render_figure(self, files: list[Path]) -> matplotlib.figure.Figure:
        """Build and return a complete standalone figure.

        Only called when ``is_multi_panel`` is True; overrides the
        compute()/render(ax) flow used by single-panel analysis plots.
        """
        raise NotImplementedError

    def prime(self, files: list[Path]) -> None:
        """Cache axis limits from the full dataset so animation frames use fixed axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            self.render(ax, self.compute(files))
            self._primed_xlim = ax.get_xlim()
            self._primed_ylims = [a.get_ylim() for a in fig.axes]
        finally:
            plt.close(fig)

    def update(self, ax: matplotlib.axes.Axes, files: list[Path]) -> None:
        """Render a prefix of snapshot files with fixed (primed) axis limits."""
        self.render(ax, self.compute(files))
        if self._primed_xlim is not None:
            ax.set_xlim(self._primed_xlim)
        if self._primed_ylims:
            ax.set_ylim(self._primed_ylims[0])
            if len(self._primed_ylims) > 1:
                # Apply twin-axis y-limits (e.g. dual-axis Ca/θ plots use twinx())
                siblings = ax.get_shared_x_axes().get_siblings(ax)
                twins = [a for a in siblings if a is not ax]
                for i, twin in enumerate(twins, start=1):
                    if i < len(self._primed_ylims):
                        twin.set_ylim(self._primed_ylims[i])


class OutputWriter(ABC):
    """Abstract base class for output writer implementations."""

    registry: ClassVar[dict[str, type[OutputWriter]]] = {}
    data_dir: str

    def __init_subclass__(cls, **kwargs: dict) -> None:
        """Register OutputWriter subclasses in the registry.

        Args:
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init_subclass__(**kwargs)

        # Skip base class itself
        if cls is OutputWriter:
            return
        key = cls.__name__.lower()
        # Prevent duplicate names
        if key in OutputWriter.registry:
            msg = f"Output writer '{key}' already registered."
            raise ValueError(msg)

        OutputWriter.registry[key] = cls

    @abstractmethod
    def save_data_step(self, iteration: int, data: dict[str, np.ndarray]) -> None:
        """Save output data for a specific iteration."""


class ConfigAdapter(ABC):
    """Converts a config source into a SimulationConfig and back.

    A subclass reads its format into the nested section layout and writes it
    back; the mapping between that layout and the config fields belongs to the
    config package (:mod:`src.config.sections`).
    """

    @abstractmethod
    def load_raw(self, source: Any) -> dict[str, Any]:
        """Parse *source* and return a flat config dict without instantiating SimulationConfig."""
        ...

    @abstractmethod
    def save(self, config: SimulationConfig, path: str) -> None:
        """Save a :class:`SimulationConfig` to *path*."""
        ...

    def load(self, source: Any) -> SimulationConfig:
        """Parse *source* and return a validated :class:`SimulationConfig`."""
        from src.config.sections import config_from_flat

        return config_from_flat(self.load_raw(source))


__all__ = [
    "AnalysisPlot",
    "BoundaryCondition",
    "BoundaryOperator",
    "CollisionOperator",
    "ConfigAdapter",
    "DifferentialOperator",
    "EosOperator",
    "EquilibriumOperator",
    "ExtDifferentialOperator",
    "ForceOperator",
    "InitialPopulationOperator",
    "InitialiserOperator",
    "MacroscopicOperator",
    "OptaxLike",
    "OutputWriter",
    "PlotOperator",
    "S",
    "SourceTermOperator",
    "StreamingOperator",
]
