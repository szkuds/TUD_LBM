"""
Unified simulation configuration for TUD-LBM.

``SimulationSetup`` is the **single public entry point** for configuring
a TUD-LBM simulation.  It flattens the physics, runner/IO, and
multiphase parameters into one typed, validated dataclass.

Usage — single-phase::

    from app_setup.simulation_setup import SimulationSetup

    setup = SimulationSetup(
        grid_shape=(128, 128),
        tau=0.8,
        nt=5000,
        collision_scheme="bgk",
    )

Usage — multiphase::

    setup = SimulationSetup(
        sim_type="multiphase",
        grid_shape=(401, 101),
        tau=0.99,
        nt=20000,
        eos="double-well",
        kappa=0.017,
        rho_l=1.0,
        rho_v=0.33,
        interface_width=4,
        bc_config={
            "left": "periodic",
            "right": "periodic",
            "top": "symmetry",
            "bottom": "wetting",
        },
        init_type="wetting",
        save_interval=2000,
    )

See Also:
    ``dev_notes/OperatorRegistry.md`` for the full developer guide.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

from app_setup.dir_config import BASE_RESULTS_DIR
from app_setup.registry import get_operators


@dataclass
class SimulationSetup:
    """Canonical, flat configuration for a TUD-LBM simulation.

    All physics, time-stepping, boundary-condition, initialisation,
    and I/O parameters live on a single object.  Multiphase-only
    fields are ``Optional`` and validated only when ``sim_type``
    is ``"multiphase"``.
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
    force_obj: Optional[List[Any]] = None

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

    # ── Extra / extensible ───────────────────────────────────────────
    extra: Dict[str, Any] = field(default_factory=dict)

    # ══════════════════════════════════════════════════════════════════
    # Validation
    # ══════════════════════════════════════════════════════════════════

    def __post_init__(self) -> None:
        # Normalise grid_shape to tuple
        if not isinstance(self.grid_shape, tuple):
            self.grid_shape = tuple(self.grid_shape)

        # Fail fast if legacy 'width' key is passed programmatically.
        if "width" in self.extra:
            raise TypeError(
                "SimulationSetup no longer accepts 'width'. "
                "Use 'interface_width' instead."
            )

        # Default bc_config to periodic on all edges
        if self.bc_config is None:
            self.bc_config = {
                "top": "periodic",
                "bottom": "periodic",
                "left": "periodic",
                "right": "periodic",
            }

        self._validate_common()

        if self.sim_type == "multiphase":
            self._validate_multiphase()

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
        valid_lattices = {"D2Q9", "D3Q19", "D3Q27"}
        if self.lattice_type not in valid_lattices:
            raise ValueError(
                f"lattice_type must be one of {valid_lattices}, "
                f"got '{self.lattice_type}'"
            )

        # tau
        if self.tau <= 0.5:
            raise ValueError(
                f"tau must be > 0.5 for stability, got {self.tau}"
            )

        # nt
        if self.nt <= 0:
            raise ValueError(f"nt must be positive, got {self.nt}")

        # collision_scheme (registry-aware)
        collision_ops = get_operators("collision_models")
        if collision_ops:
            valid_schemes = {
                name for name in collision_ops if name != "source_term"
            }
            if self.collision_scheme not in valid_schemes:
                raise ValueError(
                    f"collision_scheme must be one of "
                    f"{sorted(valid_schemes)}, got '{self.collision_scheme}'"
                )
        else:
            if self.collision_scheme not in {"bgk", "mrt"}:
                raise ValueError(
                    f"collision_scheme must be 'bgk' or 'mrt', "
                    f"got '{self.collision_scheme}'"
                )

        # k_diag required for MRT
        if self.collision_scheme == "mrt" and self.k_diag is None:
            raise ValueError(
                "k_diag must be provided when using MRT collision scheme"
            )

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
                    f"Invalid save_fields: {invalid}. "
                    f"Valid fields: {valid_fields}"
                )

    def _validate_multiphase(self) -> None:
        """Additional validation for multiphase simulations."""
        # Required multiphase fields
        for name in ("kappa", "rho_l", "rho_v", "interface_width", "eos"):
            if getattr(self, name) is None:
                raise ValueError(
                    f"'{name}' is required for multiphase simulations"
                )

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

        # EOS (registry-aware)
        macroscopic_ops = get_operators("macroscopic")
        if macroscopic_ops:
            valid_eos = {name for name in macroscopic_ops if name != "standard"}
            if self.eos not in valid_eos:
                raise ValueError(
                    f"eos must be one of {sorted(valid_eos)}, "
                    f"got '{self.eos}'"
                )
        else:
            if self.eos not in {"double-well", "carnahan-starling"}:
                raise ValueError(
                    f"eos must be 'double-well' or 'carnahan-starling', "
                    f"got '{self.eos}'"
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
        d = asdict(self)
        extra = d.pop("extra", {})
        d.update(extra)
        d["simulation_type"] = self.sim_type
        return d


    def __repr__(self) -> str:
        return (
            f"SimulationSetup(\n"
            f"  sim_type={self.sim_type!r},\n"
            f"  grid_shape={self.grid_shape!r},\n"
            f"  lattice_type={self.lattice_type!r},\n"
            f"  tau={self.tau!r},\n"
            f"  nt={self.nt!r},\n"
            f"  collision_scheme={self.collision_scheme!r},\n"
            f"  init_type={self.init_type!r},\n"
            f")"
        )


