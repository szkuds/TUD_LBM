"""Validated, serialisable simulation configuration for TUD-LBM.

:class:`SimulationConfig` is a **frozen** Python dataclass used for
parsing, validation, and serialisation. It never enters a JIT boundary.

Usage::

    from config.simulation_config import SimulationConfig

    cfg = SimulationConfig(
        grid_shape=(128, 128, 1),
        tau=0.8,
        nt=5000,
        collision_scheme="bgk",
    )
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import cast
from src.config.boundary_edges import OUT_OF_PLANE_EDGES
from src.config.boundary_edges import BoundaryEdge
from src.config.boundary_edges import build_boundary_edges
from src.config.boundary_edges import lattice_edges
from src.config.boundary_edges import pad_modes
from src.config.boundary_edges import parameter_section_key
from src.config.boundary_edges import periodic_axes
from src.config.chemical_step import ChemicalStepWall
from src.config.chemical_step import build_chemical_step_wall
from src.config.config_overview import BASE_RESULTS_DIR
from src.config.init_field import load_init_wetting
from src.config.init_field import measure_init_phase_densities
from src.config.multiphase_params import MultiphaseParams
from src.config.obstacle_mask import build_obstacle_mask
from src.config.viscosity_params import ViscosityParams
from src.config.viscosity_params import build_viscosity_params
from src.config.wetting_defaults import NEUTRAL_WETTING_CONFIG
from src.config.wetting_defaults import resolve_wetting_defaults

if TYPE_CHECKING:
    import jax.numpy as jnp

CONFIG_SECTION: str = "config_section"
ARRAY_ELIGIBLE: str = "array_eligible"
NESTED_SWEEPABLE: str = "nested_sweepable"
MIN_GRID_DIMENSIONS: int = 2
MIN_TAU_VALUE: float = 0.5


def array_field(
    *,
    default: object = dataclasses.MISSING,
    default_factory: object = dataclasses.MISSING,
    section: str | None = None,
    nested_sweepable: bool = False,
    **kwargs: object,
) -> Any:  # noqa: ANN401
    """Field factory for array-eligible SimulationConfig fields.

    Args:
        default: Default value for the field.
        default_factory: Default factory function for the field.
        section: Config section name for serialisation routing.
        nested_sweepable: If ``True``, sub-keys inside this dict field
            will also be inspected for list values during Cartesian-product
            expansion (e.g. ``gravity_force``, ``wetting_config``).
        **kwargs: Additional keyword arguments passed to the field factory.

    Returns:
        A dataclass field with array-eligible metadata.
    """
    metadata: dict[str, Any] = cast("dict[str, Any]", kwargs.pop("metadata", {}))
    metadata[ARRAY_ELIGIBLE] = True
    if nested_sweepable:
        metadata[NESTED_SWEEPABLE] = True
    if section is not None:
        metadata[CONFIG_SECTION] = section
    if default is not dataclasses.MISSING:
        return field(default=default, metadata=metadata, **kwargs)  # ty: ignore[no-matching-overload]
    if default_factory is not dataclasses.MISSING:
        return field(default_factory=default_factory, metadata=metadata, **kwargs)  # ty: ignore[no-matching-overload]
    return field(metadata=metadata, **kwargs)  # ty: ignore[no-matching-overload]


def _normalise_sequence(value: object) -> tuple[Any, ...]:
    """Ensure value is a tuple."""
    return tuple(value) if not isinstance(value, tuple) else value  # ty: ignore[invalid-argument-type]


def _first_if_list(value: object) -> object:
    """Return first element if value is a list, otherwise return value."""
    if isinstance(value, list):
        return value[0] if value else value
    return value


#: Optimiser settings of ``[hysteresis]``, filled in by ``_apply_defaults``.
_HYSTERESIS_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.01,
    "learning_rate_above": 0.05,
    "max_iterations": 50,
    "loss_tol": 1e-4,
    "trial_steps": 2,
    "carry_inactive_params": False,
    # Chemical-step runs: degrees from its bound past which a line on (or held
    # at) the post surface skips the optimiser and takes its knob's clamp limit.
    "saturation_gap": 1.0,
}
#: Every key the hysteresis operators read. ``max_iterations_above`` defaults to
#: the run's own ``max_iterations``, so it is filled in separately.
_HYSTERESIS_KEYS: frozenset[str] = frozenset(
    {"ca_advancing", "ca_receding", "max_iterations_above", *_HYSTERESIS_DEFAULTS}
)
#: Distance (lattice units) either side of a contact line at which the chemical
#: step's surfaces are probed for its advancing and receding bounds.
#: ``chemical_step_edge`` defaults to the measurement wall, so it is filled in separately.
_CHEMICAL_STEP_DEFAULTS: dict[str, Any] = {"edge_width": 1.0}


def _validate_positive(value: object, name: str) -> None:
    """Validate that value is positive."""
    if value is not None and value <= 0:  # ty: ignore[unsupported-operator]
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def _validate_nonnegative(value: object, name: str) -> None:
    """Validate that value is non-negative."""
    if value is not None and value < 0:  # ty: ignore[unsupported-operator]
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


def _valid_collision_schemes() -> set[str]:
    """Get valid collision scheme names. Returns empty set if operators not loaded."""
    try:
        import src.operators.collision  # noqa: F401
        from src.registry import get_operator_names

        return get_operator_names("collision_models")
    except (ImportError, KeyError):
        return set()  # Operators not yet loaded - skip validation


def _valid_eos() -> set[str]:
    """Get valid EOS names. Returns empty set if operators not loaded."""
    try:
        import src.operators.macroscopic.eos  # noqa: F401
        from src.registry import get_operator_names

        return get_operator_names("eos")
    except (ImportError, KeyError):
        return set()  # Operators not yet loaded - skip validation


def _valid_lattices() -> set[str]:
    """Get valid lattice types. Returns empty set if operators not loaded."""
    try:
        import src.lattice.lattice  # noqa: F401
        from src.registry import get_operator_names

        return get_operator_names("lattice")
    except (ImportError, KeyError):
        return set()  # Operators not yet loaded - skip validation


def _valid_boundary_conditions() -> set[str]:
    """Get valid boundary-condition names. Returns empty set if operators not loaded."""
    try:
        import src.operators.boundary  # noqa: F401
        from src.registry import get_operator_names

        return get_operator_names("boundary_condition")
    except (ImportError, KeyError):
        return set()  # Operators not yet loaded - skip validation


def _valid_init_types() -> set[str]:
    """Get valid init_type names. Returns empty set if operators not loaded."""
    try:
        import src.operators.initialise  # noqa: F401
        from src.registry import get_operator_names

        return get_operator_names("initialise")
    except (ImportError, KeyError):
        return set()  # Operators not yet loaded - skip validation


def _valid_obstacle_shapes() -> set[str]:
    """Get valid obstacle shape names. Returns empty set if operators not loaded."""
    try:
        import src.operators.obstacle  # noqa: F401
        from src.registry import get_operator_names

        return get_operator_names("obstacle")
    except (ImportError, KeyError):
        return set()  # Operators not yet loaded - skip validation


class _ForceSchema(NamedTuple):
    """What a force needs from its config section, declared at its registration."""

    required: tuple[str, ...]
    defaults: dict[str, Any]
    optional: tuple[str, ...]
    positive: tuple[str, ...]


def _force_schema(name: str) -> _ForceSchema:
    """Read the parameter schema a force module registered under *name*."""
    import src.operators.force  # noqa: F401
    from src.registry import get_operators

    meta: dict[str, Any] = get_operators("force")[name].metadata or {}
    return _ForceSchema(
        required=tuple(meta.get("required", ())),
        defaults=dict(meta.get("defaults", {})),
        optional=tuple(meta.get("optional", ())),
        positive=tuple(meta.get("positive", ())),
    )


@dataclass(frozen=True)
class SimulationConfig:
    """Validated, serialisable simulation configuration for TUD-LBM.

    This frozen dataclass is used for parsing, validation, and serialisation.
    It never enters a JIT boundary and serves as the configuration container
    for all simulation parameters including grid, collision, boundary conditions,
    and output settings.
    """

    # ── Simulation identity ──────────────────────────────────────
    sim_type: Literal[
        "single_phase",
        "multiphase",
        "multiphase_wetting",
        "multiphase_hysteresis",
        "multiphase_hysteresis_chemical_step",
    ] = field(
        default="single_phase",
        metadata={CONFIG_SECTION: "identity"},
    )
    simulation_name: str | None = None

    # ── Lattice & grid ───────────────────────────────────────────
    lattice_type: str = "D2Q9"
    grid_shape: tuple[int, ...] = array_field(default=(64, 64))

    # ── Time stepping ────────────────────────────────────────────
    nt: int = array_field(default=1000)
    tau: float = array_field(default=1.0)

    # ── Collision ────────────────────────────────────────────────
    collision_scheme: str = array_field(default="bgk")
    k_diag: tuple[float, ...] | None = field(default=None)

    # ── Boundary conditions (ONLY topology: which BC on which face) ──
    bc_config: dict[str, Any] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "boundary_conditions"},
    )

    # ── Interior obstacle (geometry only — no sweep support) ──────
    obstacle_config: dict[str, Any] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "obstacle"},
    )

    # ── Wetting model ──────────
    wetting_config: dict[str, Any] | None = array_field(default=None, section="wetting", nested_sweepable=True)

    # ── Hysteresis model ───────
    hysteresis_config: dict[str, Any] | None = array_field(default=None, section="hysteresis", nested_sweepable=True)

    # ── Chemical step model ───────
    chemical_step_config: dict[str, Any] | None = array_field(
        default=None, section="chemical_step", nested_sweepable=True
    )

    # ── Forces (each force is its own field, named by physics) ───
    gravity_force: dict[str, Any] | None = array_field(default=None, section="gravity_force", nested_sweepable=True)
    electric_force: dict[str, Any] | None = array_field(default=None, section="electric_force", nested_sweepable=True)
    gravity_masked_force: dict[str, Any] | None = array_field(
        default=None, section="gravity_masked_force", nested_sweepable=True
    )
    gravity_referenced_force: dict[str, Any] | None = array_field(
        default=None, section="gravity_referenced_force", nested_sweepable=True
    )
    # ── Initialisation ───────────────────────────────────────────
    init_type: str = "standard"
    init_dir: str | None = None
    initialisation: dict[str, Any] = field(
        default_factory=dict,
        metadata={CONFIG_SECTION: "initialisation"},
    )

    # ── Output / IO ──────────────────────────────────────────────
    results_dir: str = field(default=BASE_RESULTS_DIR, metadata={CONFIG_SECTION: "output"})
    save_interval: int = 0
    skip_interval: int = 0
    save_fields: list[str] | None = field(default=None, metadata={CONFIG_SECTION: "output"})
    plot_fields: list[str] | None = field(default=None, metadata={CONFIG_SECTION: "output"})
    animate_fields: list[str] | None = field(default=None, metadata={CONFIG_SECTION: "output"})
    # Plotting operators drawn on top of every field panel (e.g. ["interface"]).
    overlay_fields: list[str] | None = field(default=None, metadata={CONFIG_SECTION: "output"})
    # Interface markers to contour: "config" and/or "measured". None draws both.
    # Validated by simulation_io.analysis.interface_contour, not here.
    interface_levels: list[str] | None = field(default=None, metadata={CONFIG_SECTION: "output"})
    output_format: str | list[str] | None = field(default="numpy", metadata={CONFIG_SECTION: "output"})
    output_dir: str | None = field(default=None, metadata={CONFIG_SECTION: "output"})

    # ── Multiphase ───────────────────────────────────────────────
    eos: str | None = array_field(default=None, section="multiphase")
    kappa: float | None = array_field(default=None, section="multiphase")
    rho_l: float | None = array_field(default=None, section="multiphase")
    rho_v: float | None = array_field(default=None, section="multiphase")
    interface_width: int | None = array_field(default=None, section="multiphase")
    g: float | None = array_field(default=None, section="multiphase")
    a_eos: float | None = array_field(default=None, section="multiphase")
    b_eos: float | None = array_field(default=None, section="multiphase")
    r_eos: float | None = array_field(default=None, section="multiphase")
    t_eos: float | None = array_field(default=None, section="multiphase")
    # Shear relaxation time decoupled from the viscosity (Zhang, Guo & Wang 2022):
    # tau keeps setting the liquid viscosity, lambda_v the MRT/BGK shear rate, and
    # the equilibrium's viscous-stress term A*S makes up the difference.
    lambda_v: float | None = array_field(default=None, section="multiphase")
    # Gas viscosity nu_v = cs2*(tau_gas - 0.5); the viscosity is interpolated
    # linearly in rho between the phases. Unset means tau (uniform viscosity).
    tau_gas: float | None = array_field(default=None, section="multiphase")

    # ── Extra / extensible ───────────────────────────────────────
    extra: dict[str, Any] = field(default_factory=dict, metadata={CONFIG_SECTION: "extra"})

    # Validation

    def __post_init__(self) -> None:
        """Validate and normalise configuration after initialisation."""
        self._normalise()
        self._apply_defaults()
        self._make_grid_shape_3d()
        self._set_all_bcs()
        self._validate_common()
        if "multiphase" in self.sim_type:
            self._validate_multiphase()
        self._derive()

    def _normalise(self) -> None:
        object.__setattr__(self, "grid_shape", _normalise_sequence(self.grid_shape))
        object.__setattr__(self, "output_format", _first_if_list(self.output_format))
        if isinstance(self.output_format, str):
            object.__setattr__(self, "output_format", self.output_format.lower())
        if self.save_fields is not None and "f" not in self.save_fields:
            object.__setattr__(self, "save_fields", ["f", *self.save_fields])

    def _derive(self) -> None:
        """Fill in fields computed from other, already-validated fields.

        Runs after validation, not in ``_normalise()``: a derivation may assume
        its inputs are well formed, which is the validators' job to guarantee.
        """
        self._couple_mrt_shear_to_tau()

    def _couple_mrt_shear_to_tau(self) -> None:
        """Rewrite the shear entries of ``k_diag`` to ``1/relaxation_time``.

        The MRT shear moments set the kinematic viscosity, so left free they
        would decouple the run from ``nu = cs2*(tau - 0.5)`` — the viscosity
        every reported ``Oh``, ``La``, ``Re`` and ``Ar`` is derived from.
        With ``lambda_v`` set the shear rate is ``1/lambda_v`` and the
        viscous-stress term restores ``nu``. Deriving here rather than inside
        the collision operator keeps a saved config truthful about the rates
        its run actually used.
        """
        if self.collision_scheme != "mrt" or self.k_diag is None:
            return

        from src.operators.collision._mrt import couple_shear_to_tau

        object.__setattr__(self, "k_diag", couple_shear_to_tau(self.k_diag, self.relaxation_time))

    def _apply_defaults(self) -> None:
        self._apply_force_defaults()
        if self.save_interval == 0:
            object.__setattr__(self, "save_interval", self.nt // 10)
        if self.bc_config is None:
            object.__setattr__(
                self,
                "bc_config",
                dict.fromkeys(lattice_edges(self.lattice_type), "periodic"),
            )
        if self.hysteresis_config is not None and self.wetting_config is None:
            object.__setattr__(
                self,
                "wetting_config",
                dict(NEUTRAL_WETTING_CONFIG),
            )
        if self.hysteresis_config is not None:
            hysteresis = {**_HYSTERESIS_DEFAULTS, **self.hysteresis_config}
            hysteresis.setdefault("max_iterations_above", hysteresis["max_iterations"])
            object.__setattr__(self, "hysteresis_config", hysteresis)
        if self.chemical_step_config is not None:
            from src.operators.wetting._edge_config import first_wetting_edge

            chemical_step = {**_CHEMICAL_STEP_DEFAULTS, **self.chemical_step_config}
            chemical_step.setdefault("chemical_step_edge", first_wetting_edge(self.bc_config) or "bottom")
            object.__setattr__(self, "chemical_step_config", chemical_step)

    def _make_grid_shape_3d(self) -> None:
        """Promote grid_shape to 3D by adding a singleton z-dimension."""
        _target_dims = 3
        if len(self.grid_shape) < _target_dims:
            object.__setattr__(self, "grid_shape", self.grid_shape + (1,) * (_target_dims - len(self.grid_shape)))

    def _set_all_bcs(self) -> None:
        """Complete bc_config so the boundary builder only looks up and binds.

        Every edge of the lattice missing a BC becomes ``"periodic"``. A BC
        without its ``{edge}_{name}`` parameter section runs on the operator's
        own defaults (see :attr:`boundary_edges`).

        A two-dimensional lattice has no ``front``/``back`` face, so a periodic
        entry for one is dropped rather than carried into the overview and the
        saved ``config.toml``; run directories written before this still load.
        Any other BC there is left for :meth:`_validate_boundary_conditions`
        to reject.
        """
        if self.bc_config is None:
            return
        edges = lattice_edges(self.lattice_type)
        for edge in OUT_OF_PLANE_EDGES:
            if edge not in edges and self.bc_config.get(edge) == "periodic":
                del self.bc_config[edge]
        for edge in edges:
            if edge not in self.bc_config:
                self.bc_config[edge] = "periodic"

    def _validate_common(self) -> None:
        """Validate common simulation configuration parameters."""
        self._validate_grid_shape()
        self._validate_lattice()
        self._validate_tau()
        self._validate_time_steps()
        self._validate_collision()
        self._validate_forces()
        self._validate_init()
        self._validate_save_fields()
        self._validate_boundary_conditions()
        self._validate_obstacle()
        self._validate_hysteresis()
        self._validate_chemical_step()
        self._validate_viscosity()

    def _validate_hysteresis(self) -> None:
        """Reject ``[hysteresis]`` keys the optimiser does not read.

        The section is free-form TOML, so a misspelled key would otherwise be
        ignored without a word.
        """
        if self.hysteresis_config is None:
            return
        unknown = set(self.hysteresis_config) - _HYSTERESIS_KEYS
        if unknown:
            msg = f"Unknown [hysteresis] keys {sorted(unknown)}; allowed: {sorted(_HYSTERESIS_KEYS)}"
            raise ValueError(msg)

    def _validate_chemical_step(self) -> None:
        """Reject a chemical step that is not on the wall the contact angles are measured at.

        The wetting applicator splits a wall by surface only on ``chemical_step_edge``,
        while the hysteresis reads its contact lines off the first ``"wetting"`` edge.
        A step configured on any other edge would be ignored without a word.
        """
        if self.chemical_step_config is None:
            return
        from src.operators.wetting._edge_config import first_wetting_edge

        wall = first_wetting_edge(self.bc_config)
        edge = self.chemical_step_config["chemical_step_edge"]
        if wall is not None and edge != wall:
            msg = f"chemical_step_edge must be the wetting wall '{wall}', got '{edge}'"
            raise ValueError(msg)

    def _apply_force_defaults(self) -> None:
        """Fill every configured force section with its registered defaults."""
        for name, params in self.active_forces.items():
            object.__setattr__(self, name, {**_force_schema(name).defaults, **params})

    def _validate_forces(self) -> None:
        """Check every configured force section against the schema its module registered.

        After this, a force's ``build`` reads its section with ``params[key]``
        and never checks it again.
        """
        gravities = sorted(name for name in self.active_forces if name.startswith("gravity"))
        if len(gravities) > 1:
            msg = f"Only one gravity force can be applied, got {', '.join(gravities)}."
            raise ValueError(msg)

        for name, params in self.active_forces.items():
            schema = _force_schema(name)
            missing = [key for key in schema.required if key not in params]
            if missing:
                msg = f"[{name}] is missing required key(s): {', '.join(missing)}"
                raise ValueError(msg)
            allowed = {*schema.required, *schema.defaults, *schema.optional}
            unknown = sorted(set(params) - allowed)
            if unknown:
                msg = f"[{name}] has unknown key(s): {', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}"
                raise ValueError(msg)
            for key in schema.positive:
                _validate_positive(params.get(key), f"[{name}] {key}")

    def _validate_obstacle(self) -> None:
        """Validate interior-obstacle geometry against the grid and BC topology."""
        if self.obstacle_config is None:
            return

        self.obstacle_config.setdefault("shape", "circle")
        valid_shapes = _valid_obstacle_shapes()
        if self.obstacle_config["shape"] not in valid_shapes:
            msg = f"obstacle shape must be one of {sorted(valid_shapes)}, got '{self.obstacle_config['shape']}'"
            raise ValueError(msg)

        nx, ny, nz = self.grid_shape[:3]
        if nz > 1:
            msg = "obstacle_config only supports 2D grids (nz=1)"
            raise ValueError(msg)

        cx = self.obstacle_config.get("center_x")
        cy = self.obstacle_config.get("center_y")
        radius = self.obstacle_config.get("radius")
        if radius is None or radius <= 0:
            msg = f"obstacle radius must be positive, got {radius}"
            raise ValueError(msg)
        if cx is None or not (radius <= cx <= nx - 1 - radius):
            msg = f"obstacle x-extent [{cx - radius}, {cx + radius}] must fit within grid x in [0, {nx - 1}]"
            raise ValueError(msg)
        if cy is None or not (radius + 1 <= cy <= ny - 1 - radius - 1):
            msg = (
                f"obstacle must keep >=1 cell clearance from top/bottom walls, "
                f"got center_y={cy}, radius={radius}, ny={ny}"
            )
            raise ValueError(msg)

        assert self.bc_config is not None  # noqa: S101 - guaranteed by _apply_defaults
        if self.bc_config["left"] != "periodic" and cx - radius <= 1:
            msg = f"obstacle must keep >1 cell clearance from a non-periodic left edge, got cx={cx}, radius={radius}"
            raise ValueError(msg)
        if self.bc_config["right"] != "periodic" and cx + radius >= nx - 2:
            msg = f"obstacle must keep >1 cell clearance from a non-periodic right edge, got cx={cx}, radius={radius}"
            raise ValueError(msg)

    def _validate_boundary_conditions(self) -> None:
        """Reject an unregistered BC type, and a parameter section no edge's BC reads."""
        assert self.bc_config is not None  # noqa: S101 - guaranteed by _apply_defaults
        valid_bcs = _valid_boundary_conditions()
        edges = lattice_edges(self.lattice_type)
        for edge in OUT_OF_PLANE_EDGES:
            if edge not in edges and edge in self.bc_config:
                msg = (
                    f"bc_config['{edge}'] has no face on the {self.lattice_type} lattice, got '{self.bc_config[edge]}'"
                )
                raise ValueError(msg)
        for edge in edges:
            if self.bc_config[edge] not in valid_bcs:
                msg = f"bc_config['{edge}'] must be one of {sorted(valid_bcs)}, got '{self.bc_config[edge]}'"
                raise ValueError(msg)
        sections = {parameter_section_key(edge, self.bc_config[edge]) for edge in edges}
        for key in sorted(self.bc_config.keys() - set(edges) - sections):
            msg = f"bc_config['{key}'] is not the parameter section of any edge's boundary condition"
            raise ValueError(msg)

    def _validate_grid_shape(self) -> None:
        """Validate grid_shape dimensions."""
        if len(self.grid_shape) < MIN_GRID_DIMENSIONS:
            msg = f"grid_shape must have at least {MIN_GRID_DIMENSIONS} dimensions, got {len(self.grid_shape)}"
            raise ValueError(msg)
        if any(d <= 0 for d in self.grid_shape):
            msg = f"All grid dimensions must be positive, got {self.grid_shape}"
            raise ValueError(msg)

    def _validate_lattice(self) -> None:
        """Validate lattice_type is supported."""
        if self.lattice_type not in _valid_lattices():
            valid = _valid_lattices()
            msg = f"lattice_type must be one of {valid}, got '{self.lattice_type}'"
            raise ValueError(msg)

    def _validate_tau(self) -> None:
        """Validate tau for stability."""
        if self.tau <= MIN_TAU_VALUE:
            msg = f"tau must be > {MIN_TAU_VALUE} for stability, got {self.tau}"
            raise ValueError(msg)

    def _validate_viscosity(self) -> None:
        """Validate ``lambda_v`` / ``tau_gas`` and the positivity bound on ``A``.

        The viscous-stress parameter ``A = lambda_v - 1/2 - nu/cs2`` needs the
        density interpolation between the phases, so it is multiphase-only.
        ``|A| < lambda_v - 1/2`` keeps the scheme stable (Zhang, Guo & Wang 2022);
        ``A`` is linear in ``rho``, so checking both phase endpoints bounds it.
        """
        if self.lambda_v is None and self.tau_gas is None:
            return
        if not self.is_multiphase:
            msg = "lambda_v and tau_gas require a multiphase sim_type"
            raise ValueError(msg)
        for name in ("lambda_v", "tau_gas"):
            value = getattr(self, name)
            if value is not None and value <= MIN_TAU_VALUE:
                msg = f"{name} must be > {MIN_TAU_VALUE}, got {value}"
                raise ValueError(msg)
        margin = self.relaxation_time - MIN_TAU_VALUE
        for name, tau_phase in (("tau", self.tau), ("tau_gas", self.tau_gas or self.tau)):
            a_phase = self.relaxation_time - tau_phase
            if abs(a_phase) >= margin:
                msg = (
                    f"|A| = |lambda_v - {name}| = {abs(a_phase):.6g} must be < lambda_v - 0.5 = {margin:.6g}; "
                    f"raise lambda_v or bring {name} closer to it"
                )
                raise ValueError(msg)

    def _validate_time_steps(self) -> None:
        """Validate time stepping parameters."""
        if self.nt <= 0:
            msg = f"nt must be positive, got {self.nt}"
            raise ValueError(msg)
        _validate_nonnegative(self.save_interval, "save_interval")
        _validate_nonnegative(self.skip_interval, "skip_interval")

    def _validate_collision(self) -> None:
        """Validate collision scheme and parameters."""
        valid_schemes = _valid_collision_schemes()
        if self.collision_scheme not in valid_schemes:
            msg = f"collision_scheme must be one of {sorted(valid_schemes)}, got '{self.collision_scheme}'"
            raise ValueError(msg)

        if self.collision_scheme == "mrt" and self.k_diag is None:
            msg = "k_diag must be provided when using MRT collision scheme"
            raise ValueError(msg)

    def _validate_init(self) -> None:
        """Validate initialisation parameters."""
        valid_init_types = _valid_init_types()
        if self.init_type not in valid_init_types:
            msg = f"init_type must be one of {sorted(valid_init_types)}, got '{self.init_type}'"
            raise ValueError(msg)
        if self.init_type == "init_from_file" and self.init_dir is None:
            msg = "init_dir must be provided when init_type is 'init_from_file'"
            raise ValueError(msg)

    def _validate_save_fields(self) -> None:
        """Validate save_fields are valid."""
        if self.save_fields is not None:
            valid_fields = {"f", "rho", "u", "force", "force_ext", "pressure", "h"}
            invalid = set(self.save_fields) - valid_fields
            if invalid:
                msg = f"Invalid save_fields: {invalid}. Valid fields: {valid_fields}"
                raise ValueError(msg)

    def _validate_multiphase(self) -> None:
        required = ("kappa", "rho_l", "rho_v", "interface_width", "eos")
        for name in required:
            if getattr(self, name) is None:
                msg = f"'{name}' is required for multiphase simulations"
                raise ValueError(msg)

        _validate_positive(self.rho_l, "rho_l")
        _validate_positive(self.rho_v, "rho_v")
        if self.rho_l is not None and self.rho_v is not None and self.rho_l <= self.rho_v:
            msg = f"rho_l ({self.rho_l}) must be greater than rho_v ({self.rho_v})"
            raise ValueError(msg)
        _validate_positive(self.kappa, "kappa")
        _validate_positive(self.interface_width, "interface_width")

        valid_eos = _valid_eos()
        if self.eos not in valid_eos:
            msg = f"eos must be one of {sorted(valid_eos)}, got '{self.eos}'"
            raise ValueError(msg)

        if self.eos in {"carnahan-starling", "van-der-waals"}:
            for name in ("a_eos", "b_eos", "r_eos", "t_eos"):
                if getattr(self, name) is None:
                    msg = f"'{name}' is required when eos = '{self.eos}'"
                    raise ValueError(msg)

    @property
    def is_single_phase(self) -> bool:
        """Check if simulation is single-phase."""
        return self.sim_type == "single_phase"

    @property
    def is_multiphase(self) -> bool:
        """Check if simulation is multiphase."""
        return "multiphase" in self.sim_type

    @property
    def multiphase_params(self) -> MultiphaseParams | None:
        """The multiphase parameters, or ``None`` for a non-multiphase run.

        ``_validate_multiphase`` has already rejected any multiphase config
        missing a required field, so this is the single place the parameters
        are built and no consumer guards them again. The asserts only narrow
        the optional field types.
        """
        if not self.is_multiphase:
            return None
        assert self.eos is not None  # noqa: S101 - guaranteed by _validate_multiphase
        assert self.kappa is not None  # noqa: S101 - guaranteed by _validate_multiphase
        assert self.rho_l is not None  # noqa: S101 - guaranteed by _validate_multiphase
        assert self.rho_v is not None  # noqa: S101 - guaranteed by _validate_multiphase
        assert self.interface_width is not None  # noqa: S101 - guaranteed by _validate_multiphase
        return MultiphaseParams(
            eos=self.eos,
            kappa=self.kappa,
            rho_l=self.rho_l,
            rho_v=self.rho_v,
            interface_width=self.interface_width,
            g=self.g,
            a_eos=self.a_eos,
            b_eos=self.b_eos,
            r_eos=self.r_eos,
            t_eos=self.t_eos,
        )

    @property
    def relaxation_time(self) -> float:
        """Relaxation time of the shear moments: ``lambda_v`` when set, else ``tau``."""
        return float(self.tau if self.lambda_v is None else self.lambda_v)

    @property
    def viscosity_params(self) -> ViscosityParams | None:
        """The viscous-stress parameters, or ``None`` when the term is off.

        Off unless ``lambda_v`` or ``tau_gas`` is configured; with neither the
        shear rate is ``1/tau`` and ``A`` is identically zero.
        ``_validate_viscosity`` has already restricted these to multiphase runs.
        """
        if self.lambda_v is None and self.tau_gas is None:
            return None
        assert self.rho_l is not None  # noqa: S101 - guaranteed by _validate_multiphase
        assert self.rho_v is not None  # noqa: S101 - guaranteed by _validate_multiphase
        return build_viscosity_params(
            relaxation_time=self.relaxation_time,
            tau_liquid=float(self.tau),
            tau_gas=float(self.tau if self.tau_gas is None else self.tau_gas),
            rho_l=float(self.rho_l),
            rho_v=float(self.rho_v),
        )

    @property
    def boundary_edges(self) -> tuple[BoundaryEdge, ...]:
        """Each edge's boundary condition and its parameters, in application order.

        Static for the whole run, so resolved here rather than by the boundary
        operator package.
        """
        assert self.bc_config is not None  # noqa: S101 - guaranteed by _apply_defaults
        return build_boundary_edges(self.bc_config)

    @property
    def obstacle_mask(self) -> jnp.ndarray | None:
        """The interior-obstacle solid-cell mask; ``None`` without an obstacle."""
        nx, ny, nz = self.grid_shape[:3]
        return build_obstacle_mask(self.obstacle_config, (nx, ny, nz))

    @property
    def wetting_defaults(self) -> dict[str, float] | None:
        """The four wetting scalars by canonical name; ``None`` without wetting.

        Every hysteresis run has a (neutral) ``wetting_config`` from
        ``_apply_defaults``, so this is ``None`` only for a run with no wetting.
        """
        if self.wetting_config is None:
            return None
        return resolve_wetting_defaults(self.wetting_config)

    @property
    def pad_modes(self) -> tuple[str, str, str, str]:
        """Stencil pad mode per edge, ``(top, bottom, right, left)``, from each BC's registration."""
        assert self.bc_config is not None  # noqa: S101 - guaranteed by _apply_defaults
        return pad_modes(self.bc_config)

    @property
    def periodic_axes(self) -> tuple[bool, bool]:
        """Per-axis periodicity ``(x, y)``: both edges of the axis are periodic."""
        assert self.bc_config is not None  # noqa: S101 - guaranteed by _apply_defaults
        return periodic_axes(self.bc_config)

    @property
    def active_forces(self) -> dict[str, dict[str, Any]]:
        """The configured ``*_force`` sections, keyed by field (= registry) name.

        The single place that decides which forces a run has, so the force
        factory only looks up and binds.
        """
        return {
            f.name: params
            for f in dataclasses.fields(self)
            if f.name.endswith("_force") and (params := getattr(self, f.name)) is not None
        }

    @property
    def phase_references(self) -> tuple[float, float] | None:
        """``(rho_lo, rho_hi)`` the run's phases actually sit at, or ``None`` without two phases.

        Measured off the init field for ``init_from_file`` — an equilibrated
        field relaxes away from the prescribed coexistence densities — else the
        configured ``(rho_v, rho_l)``. ``physical_parameters`` measures the same
        field through the same reader, so the buoyancy contrast a run injects
        and the one its Bond number reports cannot diverge.
        """
        if self.rho_l is None or self.rho_v is None:
            return None
        measured = measure_init_phase_densities(self) if self.init_type == "init_from_file" else None
        return measured or (float(self.rho_v), float(self.rho_l))

    @property
    def chemical_step_wall(self) -> ChemicalStepWall | None:
        """The stepped wall the wetting applicator splits by surface; ``None`` without a step."""
        return build_chemical_step_wall(self)

    @property
    def restored_wetting(self) -> dict[str, float] | None:
        """Wetting state a hysteresis restart resumes from, or ``None`` for a fresh start.

        The hysteresis optimiser accumulates ``phi``/``d_rho`` and moves the
        contact-line anchors over a run; restarting ``init_from_file`` from one of
        its snapshots must continue from those values. Seeding them from
        ``wetting_config`` instead dropped a wall at ``phi`` ≈ 1.7 back to 1.0 on
        restart, and the contact angle ran up past its advancing bound. Only for
        hysteresis runs: a fixed-wetting run's parameters are its configuration.
        """
        if self.hysteresis_config is None or self.init_type != "init_from_file":
            return None
        return load_init_wetting(self)

    @property
    def force_enabled(self) -> bool:
        """Check if any force field is populated."""
        return bool(self.active_forces)

    # Serialisation

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary format."""
        from dataclasses import asdict

        d = asdict(self)
        extra = d.pop("extra", {})
        d.update(extra)
        d["simulation_type"] = self.sim_type
        return d

    def __repr__(self) -> str:
        """Return string representation of SimulationConfig."""
        return (
            f"SimulationConfig(\n"
            f"  sim_type={self.sim_type!r},\n"
            f"  grid_shape={self.grid_shape!r},\n"
            f"  lattice_type={self.lattice_type!r},\n"
            f"  tau={self.tau!r},\n"
            f"  nt={self.nt!r},\n"
            f"  collision_scheme={self.collision_scheme!r},\n"
            f"  init_type={self.init_type!r},\n"
            f")"
        )


def get_array_eligible_fields() -> frozenset[str]:
    """Get names of fields eligible for array expansion in configuration sweeps."""
    return frozenset(f.name for f in dataclasses.fields(SimulationConfig) if f.metadata.get(ARRAY_ELIGIBLE, False))


def get_nested_sweepable_fields() -> frozenset[str]:
    """Return field names whose dict sub-keys may carry list sweep values."""
    return frozenset(f.name for f in dataclasses.fields(SimulationConfig) if f.metadata.get(NESTED_SWEEPABLE, False))


def get_fields_for_section(section: str) -> frozenset[str]:
    """Get field names belonging to a specific configuration section."""
    return frozenset(
        f.name
        for f in dataclasses.fields(SimulationConfig)
        if f.metadata.get(CONFIG_SECTION, "simulation_type") == section
    )
