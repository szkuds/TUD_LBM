"""Validated, serialisable simulation configuration for TUD-LBM.

:class:`SimulationConfig` is a **frozen** Python dataclass used for
parsing, validation, and serialisation.  It never enters a JIT boundary.

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

# ── Serialisation section metadata ────────────────────────────────────
#
# Each dataclass field may carry a ``"config_section"`` metadata entry
# that tells adapters which top-level section the field belongs to when
# writing structured config files (TOML, YAML, JSON, …).
# Fields *without* this key default to ``"simulation_type"``.
#
# Recognised section values
# ─────────────────────────
#   "simulation_type"       → core simulation parameters (the default)
#   "multiphase"            → multiphase-specific parameters
#   "output"                → output / IO overrides
#   "boundary_conditions"   → boundary condition configuration
#   "force"                 → force definitions (array-of-tables in TOML)
#   "identity"              → special: sim_type → written as "type"
#   "extra"                 → merged into simulation_type directly
CONFIG_SECTION: str = "config_section"


def get_fields_for_section(section: str) -> frozenset[str]:
    """Return the field names whose ``config_section`` metadata equals *section*.

    Fields without ``config_section`` metadata are treated as
    ``"simulation_type"``.
    """
    return frozenset(
        f.name
        for f in dataclasses.fields(SimulationConfig)
        if f.metadata.get(CONFIG_SECTION, "simulation_type") == section
    )


def _valid_collision_schemes() -> set:
    """Return valid collision schemes from the registry."""
    from registry import ensure_registry
    from registry import get_operator_names

    ensure_registry()
    return get_operator_names("collision_models")


def _valid_eos() -> set:
    """Return valid EOS names from the registry."""
    from registry import ensure_registry
    from registry import get_operator_names

    ensure_registry()
    names = get_operator_names("macroscopic")
    return names - {"standard"}


def _valid_lattices() -> set:
    """Return valid lattice types from the registry."""
    from registry import ensure_registry
    from registry import get_operator_names

    ensure_registry()
    return get_operator_names("lattice")


@dataclass(frozen=True)
class SimulationConfig:
    """Validated, immutable simulation configuration.

    Not a JAX pytree — used only *outside* JIT for parsing,
    validation, and serialisation.

    Raises:
        ValueError: If any field value is invalid.
    """

    # ── Simulation identity ──────────────────────────────────────────
    sim_type: Literal["single_phase", "multiphase"] = field(
        default="single_phase",
        metadata={CONFIG_SECTION: "identity"},
    )
    simulation_name: str | None = None

    # ── Lattice & grid ───────────────────────────────────────────────
    lattice_type: str = "D2Q9"
    grid_shape: tuple[int, ...] = (64, 64)

    # ── Time stepping ────────────────────────────────────────────────
    nt: int = 1000
    tau: float = 1.0

    # ── Collision ────────────────────────────────────────────────────
    collision_scheme: str = "bgk"
    k_diag: tuple[float, ...] | None = None

    # ── Boundary conditions ──────────────────────────────────────────
    bc_config: dict[str, Any] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "boundary_conditions"},
    )

    # ── Force ────────────────────────────────────────────────────────
    force_enabled: bool = field(
        default=False,
        metadata={CONFIG_SECTION: "force"},
    )
    force_config: dict[str, Any] | list[dict[str, Any]] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "force"},
    )

    # ── Initialisation ───────────────────────────────────────────────
    init_type: str = "standard"
    init_dir: str | None = None

    # ── Output / IO ──────────────────────────────────────────────────
    results_dir: str = field(
        default=BASE_RESULTS_DIR,
        metadata={CONFIG_SECTION: "output"},
    )
    save_interval: int = 0  # This is set to 0 to ensure that when nothing is passed the default is nt/10
    skip_interval: int = 0
    save_fields: list[str] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "output"},
    )
    plot_fields: list[str] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "output"},
    )
    output_format: str | list[str] | None = field(
        default="numpy",
        metadata={CONFIG_SECTION: "output"},
    )

    # ── Multiphase (ignored when sim_type == "single_phase") ─────────
    eos: str | None = field(default=None, metadata={CONFIG_SECTION: "multiphase"})
    kappa: float | None = field(default=None, metadata={CONFIG_SECTION: "multiphase"})
    rho_l: float | None = field(default=None, metadata={CONFIG_SECTION: "multiphase"})
    rho_v: float | None = field(default=None, metadata={CONFIG_SECTION: "multiphase"})
    interface_width: int | None = field(default=None, metadata={CONFIG_SECTION: "multiphase"})
    bubble: bool = field(default=False, metadata={CONFIG_SECTION: "multiphase"})
    rho_ref: float | None = field(default=None, metadata={CONFIG_SECTION: "multiphase"})
    g: float | None = field(default=None, metadata={CONFIG_SECTION: "multiphase"})

    # ── Wetting / hysteresis ─────────────────────────────────────────
    wetting_config: dict[str, Any] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "boundary_conditions"},
    )
    hysteresis_config: dict[str, Any] | None = field(
        default=None,
        metadata={CONFIG_SECTION: "boundary_conditions"},
    )

    # ── Extra / extensible ───────────────────────────────────────────
    extra: dict[str, Any] = field(
        default_factory=dict,
        metadata={CONFIG_SECTION: "extra"},
    )

    # ══════════════════════════════════════════════════════════════════
    # Validation
    # ══════════════════════════════════════════════════════════════════

    def __post_init__(self) -> None:
        # frozen=True forbids normal assignment; use object.__setattr__
        # for one-time normalisation in __post_init__.

        # Compute the save_interval based on nt/10 before validation
        if self.save_interval == 0:
            object.__setattr__(self, "save_interval", self.nt // 10)

        # Normalise grid_shape to tuple
        if not isinstance(self.grid_shape, tuple):
            object.__setattr__(self, "grid_shape", tuple(self.grid_shape))

        # Normalise output_format: unwrap single-element list, lowercase
        if isinstance(self.output_format, list):
            object.__setattr__(self, "output_format", self.output_format[0])
        if isinstance(self.output_format, str):
            object.__setattr__(self, "output_format", self.output_format.lower())

        # Default bc_config to periodic on all edges
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

        self._validate_common()

        if self.sim_type == "multiphase":
            self._validate_multiphase()

    # ── Shared validation ────────────────────────────────────────────

    def _validate_common(self) -> None:
        """Validation rules shared by all simulation types."""
        # grid_shape
        if len(self.grid_shape) < 2:
            raise ValueError(
                f"grid_shape must have at least 2 dimensions, got {len(self.grid_shape)}",
            )
        if any(d <= 0 for d in self.grid_shape):
            raise ValueError(
                f"All grid dimensions must be positive, got {self.grid_shape}",
            )

        # lattice_type
        valid_lattices = _valid_lattices()
        if self.lattice_type not in valid_lattices:
            raise ValueError(
                f"lattice_type must be one of {valid_lattices}, got '{self.lattice_type}'",
            )

        # tau
        if self.tau <= 0.5:
            raise ValueError(f"tau must be > 0.5 for stability, got {self.tau}")

        # nt
        if self.nt <= 0:
            raise ValueError(f"nt must be positive, got {self.nt}")

        # collision_scheme
        valid_schemes = _valid_collision_schemes()
        if self.collision_scheme not in valid_schemes:
            raise ValueError(
                f"collision_scheme must be one of {sorted(valid_schemes)}, got '{self.collision_scheme}'",
            )

        # k_diag required for MRT
        if self.collision_scheme == "mrt" and self.k_diag is None:
            raise ValueError("k_diag must be provided when using MRT collision scheme")

        # save_interval
        if self.save_interval < 0:
            raise ValueError(
                f"save_interval must be positive, got {self.save_interval}",
            )

        # skip_interval
        if self.skip_interval < 0:
            raise ValueError(
                f"skip_interval must be non-negative, got {self.skip_interval}",
            )

        # init_dir
        if self.init_type == "init_from_file" and self.init_dir is None:
            raise ValueError(
                "init_dir must be provided when init_type is 'init_from_file'",
            )

        # save_fields
        if self.save_fields is not None:
            valid_fields = {"f", "rho", "u", "force", "force_ext", "h"}
            invalid = set(self.save_fields) - valid_fields
            if invalid:
                raise ValueError(
                    f"Invalid save_fields: {invalid}. Valid fields: {valid_fields}",
                )

    # ── Multiphase validation ────────────────────────────────────────

    def _validate_multiphase(self) -> None:
        """Additional validation for multiphase simulations."""
        for name in ("kappa", "rho_l", "rho_v", "interface_width", "eos"):
            if getattr(self, name) is None:
                raise ValueError(f"'{name}' is required for multiphase simulations")

        if self.rho_l <= 0:
            raise ValueError(f"rho_l must be positive, got {self.rho_l}")
        if self.rho_v <= 0:
            raise ValueError(f"rho_v must be positive, got {self.rho_v}")
        if self.rho_l <= self.rho_v:
            raise ValueError(
                f"rho_l ({self.rho_l}) must be greater than rho_v ({self.rho_v})",
            )
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.interface_width <= 0:
            raise ValueError(
                f"interface_width must be positive, got {self.interface_width}",
            )

        # EOS
        valid_eos = _valid_eos()
        if self.eos not in valid_eos:
            raise ValueError(
                f"eos must be one of {sorted(valid_eos)}, got '{self.eos}'",
            )

    # ══════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════

    @property
    def is_single_phase(self) -> bool:
        """Whether this is a single-phase simulation."""
        return self.sim_type == "single_phase"

    @property
    def is_multiphase(self) -> bool:
        """Whether this is a multiphase simulation."""
        return self.sim_type == "multiphase"

    # ══════════════════════════════════════════════════════════════════
    # Serialisation
    # ══════════════════════════════════════════════════════════════════

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for logging / saving.

        Merges ``extra`` into the top-level dict and adds a
        ``simulation_type`` key.
        """
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
