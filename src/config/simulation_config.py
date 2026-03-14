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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from config.dir_config import BASE_RESULTS_DIR

# Fallback sets used when registry is not yet populated (e.g. config-only tests).
_FALLBACK_COLLISION_SCHEMES = {"bgk", "mrt"}
_FALLBACK_EOS = {"double-well", "carnahan-starling"}
_FALLBACK_LATTICES = {"D2Q9", "D3Q19", "D3Q27"}


def _valid_collision_schemes() -> set:
    """Return valid collision schemes from the registry, with fallback."""
    try:
        from registry import get_operator_names

        names = get_operator_names("collision_models")
        if names:
            return names
    except Exception:
        pass
    return _FALLBACK_COLLISION_SCHEMES


def _valid_eos() -> set:
    """Return valid EOS names from the registry, with fallback."""
    try:
        from registry import get_operator_names

        names = get_operator_names("macroscopic")
        # EOS names are the multiphase macroscopic operator names (exclude 'standard')
        eos_names = names - {"standard"}
        if eos_names:
            return eos_names
    except Exception:
        pass
    return _FALLBACK_EOS


def _valid_lattices() -> set:
    """Return valid lattice types from the registry, with fallback."""
    try:
        from registry import get_operator_names

        names = get_operator_names("lattice")
        if names:
            return names
    except Exception:
        pass
    return _FALLBACK_LATTICES


@dataclass(frozen=True)
class SimulationConfig:
    """Validated, immutable simulation configuration.

    Not a JAX pytree — used only *outside* JIT for parsing,
    validation, and serialisation.

    Raises:
        ValueError: If any field value is invalid.
    """

    # ── Simulation identity ──────────────────────────────────────────
    sim_type: Literal["single_phase", "multiphase"] = "single_phase"
    simulation_name: Optional[str] = None

    # ── Lattice & grid ───────────────────────────────────────────────
    lattice_type: str = "D2Q9"
    grid_shape: Tuple[int, ...] = (64, 64)

    # ── Time stepping ────────────────────────────────────────────────
    nt: int = 1000
    tau: float = 1.0

    # ── Collision ────────────────────────────────────────────────────
    collision_scheme: str = "bgk"
    k_diag: Optional[Tuple[float, ...]] = None

    # ── Boundary conditions ──────────────────────────────────────────
    bc_config: Optional[Dict[str, Any]] = None

    # ── Force ────────────────────────────────────────────────────────
    force_enabled: bool = False
    force_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None

    # ── Initialisation ───────────────────────────────────────────────
    init_type: str = "standard"
    init_dir: Optional[str] = None

    # ── Output / IO ──────────────────────────────────────────────────
    results_dir: str = BASE_RESULTS_DIR
    save_interval: int = 100
    skip_interval: int = 0
    save_fields: Optional[List[str]] = None

    # ── Multiphase (ignored when sim_type == "single_phase") ─────────
    eos: Optional[str] = None
    kappa: Optional[float] = None
    rho_l: Optional[float] = None
    rho_v: Optional[float] = None
    interface_width: Optional[int] = None
    bubble: bool = False
    rho_ref: Optional[float] = None
    g: Optional[float] = None

    # ── Wetting / hysteresis ─────────────────────────────────────────
    wetting_config: Optional[Dict[str, Any]] = None
    hysteresis_config: Optional[Dict[str, Any]] = None

    # ── Extra / extensible ───────────────────────────────────────────
    extra: Dict[str, Any] = field(default_factory=dict)

    # ══════════════════════════════════════════════════════════════════
    # Validation
    # ══════════════════════════════════════════════════════════════════

    def __post_init__(self) -> None:
        # frozen=True forbids normal assignment; use object.__setattr__
        # for one-time normalisation in __post_init__.

        # Normalise grid_shape to tuple
        if not isinstance(self.grid_shape, tuple):
            object.__setattr__(self, "grid_shape", tuple(self.grid_shape))

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
                f"grid_shape must have at least 2 dimensions, "
                f"got {len(self.grid_shape)}"
            )
        if any(d <= 0 for d in self.grid_shape):
            raise ValueError(
                f"All grid dimensions must be positive, got {self.grid_shape}"
            )

        # lattice_type
        valid_lattices = _valid_lattices()
        if self.lattice_type not in valid_lattices:
            raise ValueError(
                f"lattice_type must be one of {valid_lattices}, "
                f"got '{self.lattice_type}'"
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
                f"collision_scheme must be one of "
                f"{sorted(valid_schemes)}, got '{self.collision_scheme}'"
            )

        # k_diag required for MRT
        if self.collision_scheme == "mrt" and self.k_diag is None:
            raise ValueError("k_diag must be provided when using MRT collision scheme")

        # save_interval
        if self.save_interval <= 0:
            raise ValueError(
                f"save_interval must be positive, got {self.save_interval}"
            )

        # skip_interval
        if self.skip_interval < 0:
            raise ValueError(
                f"skip_interval must be non-negative, got {self.skip_interval}"
            )

        # init_dir
        if self.init_type == "init_from_file" and self.init_dir is None:
            raise ValueError(
                "init_dir must be provided when init_type is 'init_from_file'"
            )

        # save_fields
        if self.save_fields is not None:
            valid_fields = {"f", "rho", "u", "force", "force_ext", "h"}
            invalid = set(self.save_fields) - valid_fields
            if invalid:
                raise ValueError(
                    f"Invalid save_fields: {invalid}. " f"Valid fields: {valid_fields}"
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
                f"rho_l ({self.rho_l}) must be greater than rho_v ({self.rho_v})"
            )
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.interface_width <= 0:
            raise ValueError(
                f"interface_width must be positive, got {self.interface_width}"
            )

        # EOS
        valid_eos = _valid_eos()
        if self.eos not in valid_eos:
            raise ValueError(
                f"eos must be one of {sorted(valid_eos)}, " f"got '{self.eos}'"
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

    def to_dict(self) -> Dict[str, Any]:
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
