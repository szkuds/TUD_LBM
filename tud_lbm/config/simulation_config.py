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
from typing import Any
from typing import Literal
from tud_lbm.config.config_overview import BASE_RESULTS_DIR

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
) -> field:
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
    metadata = dict(kwargs.pop("metadata", {}))
    metadata[ARRAY_ELIGIBLE] = True
    if nested_sweepable:
        metadata[NESTED_SWEEPABLE] = True
    if section is not None:
        metadata[CONFIG_SECTION] = section
    return field(default=default, default_factory=default_factory, metadata=metadata, **kwargs)


def _normalize_sequence(value: object) -> tuple[Any, ...]:
    """Ensure value is a tuple."""
    return tuple(value) if not isinstance(value, tuple) else value  # type: ignore[arg-type]


def _first_if_list(value: object) -> object:
    """Return first element if value is a list, otherwise return value."""
    if isinstance(value, list):
        return value[0] if value else value
    return value


def _validate_positive(value: object, name: str) -> None:
    """Validate that value is positive."""
    if value is not None and value <= 0:  # type: ignore[operator]
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def _validate_nonnegative(value: object, name: str) -> None:
    """Validate that value is non-negative."""
    if value is not None and value < 0:  # type: ignore[operator]
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


def _valid_collision_schemes() -> set[str]:
    """Get valid collision scheme names. Returns empty set if operators not loaded."""
    try:
        import tud_lbm.operators.collision  # noqa: F401
        from tud_lbm.registry import get_operator_names

        return get_operator_names("collision_models")
    except (ImportError, KeyError):
        return set()  # Operators not yet loaded - skip validation


def _valid_eos() -> set[str]:
    """Get valid EOS names. Returns empty set if operators not loaded."""
    try:
        import tud_lbm.operators.macroscopic.eos  # noqa: F401
        from tud_lbm.registry import get_operator_names

        return get_operator_names("eos")
    except (ImportError, KeyError):
        return set()  # Operators not yet loaded - skip validation


def _valid_lattices() -> set[str]:
    """Get valid lattice types. Returns empty set if operators not loaded."""
    try:
        import tud_lbm.lattice.lattice  # noqa: F401
        from tud_lbm.registry import get_operator_names

        return get_operator_names("lattice")
    except (ImportError, KeyError):
        return set()  # Operators not yet loaded - skip validation


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

    # ── Extra / extensible ───────────────────────────────────────
    extra: dict[str, Any] = field(default_factory=dict, metadata={CONFIG_SECTION: "extra"})

    # Validation

    def __post_init__(self) -> None:
        """Validate and normalize configuration after initialization."""
        self._normalize()
        self._apply_defaults()
        self._make_grid_shape_3d()
        self._set_all_bcs()
        self._validate_common()
        if "multiphase" in self.sim_type:
            self._validate_multiphase()

    def _normalize(self) -> None:
        object.__setattr__(self, "grid_shape", _normalize_sequence(self.grid_shape))
        object.__setattr__(self, "output_format", _first_if_list(self.output_format))
        if isinstance(self.output_format, str):
            object.__setattr__(self, "output_format", self.output_format.lower())

    def _apply_defaults(self) -> None:
        if self.save_interval == 0:
            object.__setattr__(self, "save_interval", self.nt // 10)
        if self.bc_config is None:
            object.__setattr__(
                self,
                "bc_config",
                {
                    "top": "periodic",
                    "bottom": "periodic",
                    "left": "periodic",
                    "right": "periodic",
                    "front": "periodic",
                    "back": "periodic",
                },
            )
        if self.hysteresis_config is not None and self.wetting_config is None:
            object.__setattr__(
                self,
                "wetting_config",
                {
                    "phi_left": 1.0,
                    "phi_right": 1.0,
                    "d_rho_left": 0.0,
                    "d_rho_right": 0.0,
                },
            )

    def _make_grid_shape_3d(self) -> None:
        """Promote grid_shape to 3D by adding a singleton z-dimension."""
        _target_dims = 3
        if len(self.grid_shape) < _target_dims:
            object.__setattr__(self, "grid_shape", self.grid_shape + (1,) * (_target_dims - len(self.grid_shape)))

    def _set_all_bcs(self) -> None:
        """Set missing BCs in bc_config to 'periodic'."""
        for edge in ("top", "bottom", "left", "right", "front", "back"):
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

    def _validate_forces(self) -> None:
        """Validate force configuration consistency."""
        if self.gravity_force is not None and self.gravity_masked_force is not None:
            msg = "Only one gravity force can be applied: set either gravity_force or gravity_masked_force, not both."
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
        """Validate initialization parameters."""
        if self.init_type == "init_from_file" and self.init_dir is None:
            msg = "init_dir must be provided when init_type is 'init_from_file'"
            raise ValueError(msg)

    def _validate_save_fields(self) -> None:
        """Validate save_fields are valid."""
        if self.save_fields is not None:
            valid_fields = {"f", "rho", "u", "force", "force_ext", "h"}
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

        if self.eos == "carnahan-starling":
            for name in ("a_eos", "b_eos", "r_eos", "t_eos"):
                if getattr(self, name) is None:
                    msg = f"'{name}' is required when eos = 'carnahan-starling'"
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
    def force_enabled(self) -> bool:
        """Check if any force field is populated."""
        return any(getattr(self, f.name) is not None for f in dataclasses.fields(self) if f.name.endswith("_force"))

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
