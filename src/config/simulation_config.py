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
from src.config._checks import registered_names
from src.config._checks import validate_nonnegative
from src.config.boundary import complete_boundary_conditions
from src.config.boundary import is_three_dimensional
from src.config.boundary import validate_boundary_conditions
from src.config.config_overview import BASE_RESULTS_DIR
from src.config.defaults import DEFAULT_EQUILIBRIUM
from src.config.equilibrium import validate_equilibrium
from src.config.forces import apply_force_defaults
from src.config.forces import validate_forces
from src.config.multiphase import complete_multiphase
from src.config.multiphase import validate_multiphase
from src.config.obstacle import complete_obstacle
from src.config.obstacle import validate_obstacle
from src.config.wetting import complete_chemical_step
from src.config.wetting import complete_hysteresis
from src.config.wetting import complete_wetting
from src.config.wetting import validate_chemical_step
from src.config.wetting import validate_hysteresis
from src.config.wetting import validate_wetting

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
            expansion (e.g. ``forces``, ``wetting``).
        **kwargs: Additional keyword arguments passed to the field factory.

    Returns:
        A dataclass field with array-eligible metadata.
    """
    metadata: dict[str, Any] = {}
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


@dataclass(frozen=True)
class SimulationConfig:
    """Validated, serialisable simulation configuration for TUD-LBM.

    This frozen dataclass is used for parsing, validation, and serialisation.
    It never enters a JIT boundary and serves as the configuration container
    for all simulation parameters including grid, collision, boundary conditions,
    and output settings.
    """

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

    # ── Equilibrium (its source term is the pair it registered) ──
    equilibrium: str | None = None

    # ── Sections: one field per TOML section, each owned by its config module ──
    boundary_conditions: dict[str, Any] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "boundary_conditions"},
    )
    obstacle: dict[str, Any] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "obstacle"},
    )
    # Its presence is what makes a run multiphase.
    multiphase: dict[str, Any] | None = array_field(default=None, section="multiphase", nested_sweepable=True)
    wetting: dict[str, Any] | None = array_field(default=None, section="wetting", nested_sweepable=True)
    hysteresis: dict[str, Any] | None = array_field(default=None, section="hysteresis", nested_sweepable=True)
    chemical_step: dict[str, Any] | None = array_field(default=None, section="chemical_step", nested_sweepable=True)
    # Every [*_force] section, under the section's name.
    forces: dict[str, dict[str, Any]] = array_field(default_factory=dict, section="forces", nested_sweepable=True)

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

    # ── Extra / extensible ───────────────────────────────────────
    extra: dict[str, Any] = field(default_factory=dict, metadata={CONFIG_SECTION: "extra"})

    # Validation

    def __post_init__(self) -> None:
        """Validate and normalise configuration after initialisation."""
        self._normalise()
        self._apply_defaults()
        self._make_grid_shape_3d()
        self._validate()
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
        its inputs are well-formed, which is the validators' job to guarantee.
        """
        self._couple_mrt_shear_to_tau()

    def _couple_mrt_shear_to_tau(self) -> None:
        """Rewrite the shear entries of ``k_diag`` to ``1/tau``.

        The MRT shear moments set the kinematic viscosity, so left free they
        would decouple the run from ``nu = cs2*(tau - 0.5)`` — the viscosity
        every reported ``Oh``, ``La``, ``Re`` and ``Ar`` is derived from.
        Deriving here rather than inside
        the collision operator keeps a saved config truthful about the rates
        its run actually used.
        """
        if self.collision_scheme != "mrt" or self.k_diag is None:
            return

        from src.operators.collision._mrt import couple_shear_to_tau

        object.__setattr__(self, "k_diag", couple_shear_to_tau(self.k_diag, float(self.tau)))

    def _apply_defaults(self) -> None:
        phase = "single_phase" if self.multiphase is None else "multiphase"
        edges = complete_boundary_conditions(self.boundary_conditions, self.lattice_type)
        equilibrium = self.equilibrium or DEFAULT_EQUILIBRIUM[phase]
        defaults: dict[str, Any] = {
            "boundary_conditions": edges,
            "multiphase": complete_multiphase(self.multiphase, equilibrium),
            "obstacle": complete_obstacle(self.obstacle),
            "wetting": complete_wetting(self.wetting, self.hysteresis),
            "hysteresis": complete_hysteresis(self.hysteresis),
            "chemical_step": complete_chemical_step(
                self.chemical_step, edges, self.initialisation, int(self.grid_shape[0])
            ),
            "forces": apply_force_defaults(self.forces),
            "equilibrium": equilibrium,
            "save_interval": self.save_interval or self.nt // 10,
        }
        for name, value in defaults.items():
            object.__setattr__(self, name, value)

    def _make_grid_shape_3d(self) -> None:
        """Promote grid_shape to 3D by adding a singleton z-dimension."""
        _target_dims = 3
        if len(self.grid_shape) < _target_dims:
            object.__setattr__(self, "grid_shape", self.grid_shape + (1,) * (_target_dims - len(self.grid_shape)))

    def _validate(self) -> None:
        """Run the core checks, then each section's own."""
        self._validate_grid_shape()
        self._validate_lattice()
        self._validate_tau()
        self._validate_time_steps()
        self._validate_collision()
        validate_forces(self)
        self._validate_init()
        self._validate_save_fields()
        validate_boundary_conditions(self)
        validate_obstacle(self)
        validate_wetting(self)
        validate_hysteresis(self)
        validate_chemical_step(self)
        validate_multiphase(self)
        self._validate_dimension()

    def _validate_dimension(self) -> None:
        """Keep the grid on its lattice, and refuse what has no 3D form: bulk runs only there.

        A 3D run of any of these would otherwise fail deep in a trace, or use
        only the ``z = 0`` plane without a word.
        """
        if not is_three_dimensional(self.lattice_type):
            if self.grid_shape[2] != 1:
                msg = f"{self.lattice_type} is two-dimensional, got nz={self.grid_shape[2]}; use a 3D lattice"
                raise ValueError(msg)
            return
        edges = self.boundary_conditions or {}
        unsupported = {
            "wetting": self.wetting is not None or "wetting" in edges.values(),
            "hysteresis": self.hysteresis is not None,
            "chemical_step": self.chemical_step is not None,
            "electric_force": "electric_force" in self.forces,
            "MRT collision": self.collision_scheme == "mrt",
            "obstacle": self.obstacle is not None,
        }
        refused = [name for name, used in unsupported.items() if used]
        if refused:
            msg = f"{', '.join(refused)} not supported on the 3D lattice {self.lattice_type}"
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
        valid = registered_names("src.lattice.lattice", "lattice")
        if self.lattice_type not in valid:
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
        validate_nonnegative(self.save_interval, "save_interval")
        validate_nonnegative(self.skip_interval, "skip_interval")

    def _validate_collision(self) -> None:
        """Validate collision scheme and parameters."""
        valid_schemes = registered_names("src.operators.collision", "collision_models")
        if self.collision_scheme not in valid_schemes:
            msg = f"collision_scheme must be one of {sorted(valid_schemes)}, got '{self.collision_scheme}'"
            raise ValueError(msg)

        if self.collision_scheme == "mrt" and self.k_diag is None:
            msg = "k_diag must be provided when using MRT collision scheme"
            raise ValueError(msg)
        assert self.equilibrium is not None  # noqa: S101 - _apply_defaults fills it in
        validate_equilibrium(self.equilibrium, multiphase=self.multiphase is not None)

    def _validate_init(self) -> None:
        """Validate initialisation parameters."""
        valid_init_types = registered_names("src.operators.initialise", "initialise")
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

    # Serialisation

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary format."""
        from dataclasses import asdict

        d = asdict(self)
        extra = d.pop("extra", {})
        d.update(extra)
        return d

    def __repr__(self) -> str:
        """Return string representation of SimulationConfig."""
        return (
            f"SimulationConfig(\n"
            f"  multiphase={self.multiphase!r},\n"
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
