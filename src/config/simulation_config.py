"""Validated, serialisable simulation configuration for TUD-LBM.

:class:`SimulationConfig` is a **frozen** Python dataclass used for
parsing, validation, and serialisation. It never enters a JIT boundary.

Usage::

    from config.simulation_config import SimulationConfig

    cfg = SimulationConfig(
        grid_shape=(128, 128),
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
from config.dir_config import BASE_RESULTS_DIR

CONFIG_SECTION: str = "config_section"
ARRAY_ELIGIBLE: str = "array_eligible"
NESTED_SWEEPABLE: str = "nested_sweepable"


def array_field(
    *,
    default=dataclasses.MISSING,
    default_factory=dataclasses.MISSING,
    section: str | None = None,
    nested_sweepable: bool = False,
    **kwargs,
):
    """Field factory for array-eligible SimulationConfig fields.

    Args:
        section: Config section name for serialisation routing.
        nested_sweepable: If ``True``, sub-keys inside this dict field
            will also be inspected for list values during Cartesian-product
            expansion (e.g. ``gravity_force``, ``wetting_config``).
    """
    metadata = dict(kwargs.pop("metadata", {}))
    metadata[ARRAY_ELIGIBLE] = True
    if nested_sweepable:
        metadata[NESTED_SWEEPABLE] = True
    if section is not None:
        metadata[CONFIG_SECTION] = section
    return field(default=default, default_factory=default_factory, metadata=metadata, **kwargs)


def _normalize_sequence(value):
    return tuple(value) if not isinstance(value, tuple) else value


def _first_if_list(value):
    if isinstance(value, list):
        return value[0] if value else value
    return value


def _validate_positive(value, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_nonnegative(value, name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _valid_collision_schemes() -> set[str]:
    import operators.collision  # noqa: F401
    from registry import get_operator_names

    return get_operator_names("collision_models")


def _valid_eos() -> set[str]:
    import operators.macroscopic  # noqa: F401
    from registry import get_operator_names

    return get_operator_names("macroscopic") - {"standard"}


def _valid_lattices() -> set[str]:
    import setup.lattice  # noqa: F401
    from registry import get_operator_names

    return get_operator_names("lattice")


@dataclass(frozen=True)
class SimulationConfig:
    # ── Simulation identity ──────────────────────────────────────
    sim_type: Literal["single_phase", "multiphase"] = field(
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
    k_diag: tuple[float, ...] | None = array_field(default=None)

    # ── Boundary conditions (ONLY topology: which BC on which face) ──
    bc_config: dict[str, Any] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "boundary_conditions"},
    )

    # ── Wetting model (promoted to first-class section) ──────────
    wetting_config: dict[str, Any] | None = array_field(default=None, section="wetting", nested_sweepable=True)

    # ── Hysteresis model (promoted to first-class section) ───────
    hysteresis_config: dict[str, Any] | None = array_field(default=None, section="hysteresis", nested_sweepable=True)

    # ── Forces (each force is its own field, named by physics) ───
    gravity_force: dict[str, Any] | None = array_field(default=None, section="gravity_force", nested_sweepable=True)
    electric_force: dict[str, Any] | None = array_field(default=None, section="electric_force", nested_sweepable=True)
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
    output_format: str | list[str] | None = field(default="numpy", metadata={CONFIG_SECTION: "output"})
    output_dir: str | None = field(default=None, metadata={CONFIG_SECTION: "output"})

    # ── Multiphase ───────────────────────────────────────────────
    eos: str | None = array_field(default=None, section="multiphase")
    kappa: float | None = array_field(default=None, section="multiphase")
    rho_l: float | None = array_field(default=None, section="multiphase")
    rho_v: float | None = array_field(default=None, section="multiphase")
    interface_width: int | None = array_field(default=None, section="multiphase")
    g: float | None = array_field(default=None, section="multiphase")

    # ── Extra / extensible ───────────────────────────────────────
    extra: dict[str, Any] = field(default_factory=dict, metadata={CONFIG_SECTION: "extra"})

    # Validation

    def __post_init__(self) -> None:
        self._normalize()
        self._apply_defaults()
        self._validate_common()
        if self.sim_type == "multiphase":
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
                },
            )

    def _validate_common(self) -> None:
        if len(self.grid_shape) < 2:
            raise ValueError(f"grid_shape must have at least 2 dimensions, got {len(self.grid_shape)}")
        if any(d <= 0 for d in self.grid_shape):
            raise ValueError(f"All grid dimensions must be positive, got {self.grid_shape}")

        if self.lattice_type not in _valid_lattices():
            valid = _valid_lattices()
            raise ValueError(f"lattice_type must be one of {valid}, got '{self.lattice_type}'")

        if self.tau <= 0.5:
            raise ValueError(f"tau must be > 0.5 for stability, got {self.tau}")

        if self.nt <= 0:
            raise ValueError(f"nt must be positive, got {self.nt}")

        valid_schemes = _valid_collision_schemes()
        if self.collision_scheme not in valid_schemes:
            raise ValueError(f"collision_scheme must be one of {sorted(valid_schemes)}, got '{self.collision_scheme}'")

        if self.collision_scheme == "mrt" and self.k_diag is None:
            raise ValueError("k_diag must be provided when using MRT collision scheme")

        _validate_nonnegative(self.save_interval, "save_interval")
        _validate_nonnegative(self.skip_interval, "skip_interval")

        if self.init_type == "init_from_file" and self.init_dir is None:
            raise ValueError("init_dir must be provided when init_type is 'init_from_file'")

        if self.save_fields is not None:
            valid_fields = {"f", "rho", "u", "force", "force_ext", "h"}
            invalid = set(self.save_fields) - valid_fields
            if invalid:
                raise ValueError(f"Invalid save_fields: {invalid}. Valid fields: {valid_fields}")

    def _validate_multiphase(self) -> None:
        required = ("kappa", "rho_l", "rho_v", "interface_width", "eos")
        for name in required:
            if getattr(self, name) is None:
                raise ValueError(f"'{name}' is required for multiphase simulations")

        _validate_positive(self.rho_l, "rho_l")
        _validate_positive(self.rho_v, "rho_v")
        if self.rho_l is not None and self.rho_v is not None and self.rho_l <= self.rho_v:
            raise ValueError(f"rho_l ({self.rho_l}) must be greater than rho_v ({self.rho_v})")
        _validate_positive(self.kappa, "kappa")
        _validate_positive(self.interface_width, "interface_width")

        valid_eos = _valid_eos()
        if self.eos not in valid_eos:
            raise ValueError(f"eos must be one of {sorted(valid_eos)}, got '{self.eos}'")

    @property
    def is_single_phase(self) -> bool:
        return self.sim_type == "single_phase"

    @property
    def is_multiphase(self) -> bool:
        return self.sim_type == "multiphase"

    @property
    def force_enabled(self) -> bool:
        """True if any ``*_force`` field is populated."""
        return any(getattr(self, f.name) is not None for f in dataclasses.fields(self) if f.name.endswith("_force"))

    # Serialisation

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        d = asdict(self)
        extra = d.pop("extra", {})
        d.update(extra)
        d["simulation_type"] = self.sim_type
        return d

    def __repr__(self) -> str:
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
    return frozenset(f.name for f in dataclasses.fields(SimulationConfig) if f.metadata.get(ARRAY_ELIGIBLE, False))


def get_nested_sweepable_fields() -> frozenset[str]:
    """Return field names whose dict sub-keys may carry list sweep values."""
    return frozenset(f.name for f in dataclasses.fields(SimulationConfig) if f.metadata.get(NESTED_SWEEPABLE, False))


def get_fields_for_section(section: str) -> frozenset[str]:
    return frozenset(
        f.name
        for f in dataclasses.fields(SimulationConfig)
        if f.metadata.get(CONFIG_SECTION, "simulation_type") == section
    )
