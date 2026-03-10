"""
Configuration dataclasses for LBM simulations.

Provides validated, type-safe configuration objects that replace
the sprawling keyword arguments in simulation constructors.

Each simulation type has its own config class with:
- Type annotations for IDE support
- Default values matching previous behavior
- __post_init__ validation for early error detection

Architecture:
    BaseSimulationConfig
    ├── SinglePhaseConfig
    └── MultiphaseConfig

    RunnerConfig          (I/O, saving, initialisation)

    SimulationBundle      (top-level composite returned by file adapters)
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from config.dir_config import BASE_RESULTS_DIR
from registry import get_operators



@dataclass
class BaseSimulationConfig:
    """
    Base configuration shared by all simulation types.

    Attributes:
        grid_shape: Shape of the simulation grid (nx, ny) or (nx, ny, nz).
        lattice_type: Lattice velocity set, e.g. "D2Q9", "D3Q19".
        tau: Relaxation time parameter.
        nt: Number of timesteps to run.
    """
    grid_shape: Tuple[int, ...]
    lattice_type: str = "D2Q9"
    tau: float = 1.0
    nt: int = 1000

    def __post_init__(self):
        # Validate grid_shape
        if not isinstance(self.grid_shape, tuple):
            self.grid_shape = tuple(self.grid_shape)
        if len(self.grid_shape) < 2:
            raise ValueError(f"grid_shape must have at least 2 dimensions, got {len(self.grid_shape)}")
        if any(d <= 0 for d in self.grid_shape):
            raise ValueError(f"All grid dimensions must be positive, got {self.grid_shape}")

        # Validate lattice_type
        valid_lattices = {"D2Q9", "D3Q19", "D3Q27"}
        if self.lattice_type not in valid_lattices:
            raise ValueError(f"lattice_type must be one of {valid_lattices}, got '{self.lattice_type}'")

        # Validate tau
        if self.tau <= 0.5:
            raise ValueError(f"tau must be > 0.5 for stability, got {self.tau}")

        # Validate nt
        if self.nt <= 0:
            raise ValueError(f"nt must be positive, got {self.nt}")


@dataclass
class SinglePhaseConfig(BaseSimulationConfig):
    """
    Configuration for single-phase LBM simulations.

    Attributes:
        force_enabled: Whether external forcing is enabled.
        force_obj: Force object(s) for external forcing.
        bc_config: Boundary condition configuration dict.
        collision_scheme: Collision operator type ("bgk" or "mrt").
        k_diag: MRT relaxation rates diagonal (for collision_scheme="mrt").
    """
    force_enabled: bool = False
    force_obj: Optional[Any] = None
    bc_config: Optional[Dict[str, Any]] = None
    collision_scheme: str = "bgk"
    k_diag: Optional[Tuple[float, ...]] = None

    # Extra kwargs for extensibility
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        # Validate collision_scheme using the registry (dynamically populated)
        collision_ops = get_operators("collision")
        if collision_ops:
            # Filter to only actual collision operators (exclude source_term etc.)
            valid_schemes = {name for name, entry in collision_ops.items()
                            if name not in ("source_term",)}
            if self.collision_scheme not in valid_schemes:
                raise ValueError(
                    f"collision_scheme must be one of {sorted(valid_schemes)}, "
                    f"got '{self.collision_scheme}'"
                )
        else:
            # Fallback when registry has not been populated yet
            valid_schemes = {"bgk", "mrt"}
            if self.collision_scheme not in valid_schemes:
                raise ValueError(f"collision_scheme must be one of {valid_schemes}, got '{self.collision_scheme}'")

        # Validate k_diag is provided for MRT
        if self.collision_scheme == "mrt" and self.k_diag is None:
            raise ValueError("k_diag must be provided when using MRT collision scheme")


@dataclass
class MultiphaseConfig(BaseSimulationConfig):
    """
    Configuration for multiphase (two-phase) LBM simulations.

    Attributes:
        kappa: Surface tension parameter.
        rho_l: Liquid phase density.
        rho_v: Vapor phase density.
        interface_width: Diffuse interface width in lattice units.
        eos: Equation of state ("double-well" or "carnahan-starling").
        force_enabled: Whether external forcing is enabled.
        force_obj: Force object(s) for external forcing (list of force objects).
        bc_config: Boundary condition configuration dict.
        collision_scheme: Collision operator type ("bgk" or "mrt").
        k_diag: MRT relaxation rates diagonal (for collision_scheme="mrt").
        bubble: Whether initializing a bubble (vs droplet).
        rho_ref: Reference density for bubble initialization.
        g: Gravitational acceleration for bubble initialization.
    """
    kappa: float = 0.1
    rho_l: float = 1.0
    rho_v: float = 0.1
    interface_width: int = 4
    eos: str = "double-well"
    force_enabled: bool = False
    force_obj: Optional[List[Any]] = None
    bc_config: Optional[Dict[str, Any]] = None
    collision_scheme: str = "bgk"
    k_diag: Optional[Tuple[float, ...]] = None
    bubble: bool = False
    rho_ref: Optional[float] = None
    g: Optional[float] = None

    # Extra kwargs for extensibility
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        # Validate densities
        if self.rho_l <= 0:
            raise ValueError(f"rho_l must be positive, got {self.rho_l}")
        if self.rho_v <= 0:
            raise ValueError(f"rho_v must be positive, got {self.rho_v}")
        if self.rho_l <= self.rho_v:
            raise ValueError(f"rho_l ({self.rho_l}) must be greater than rho_v ({self.rho_v})")

        # Validate kappa
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")

        # Validate interface_width
        if self.interface_width <= 0:
            raise ValueError(f"interface_width must be positive, got {self.interface_width}")

        # Validate eos using the registry (dynamically populated)
        macroscopic_ops = get_operators("macroscopic")
        if macroscopic_ops:
            # Exclude "standard" since it's not a multiphase EOS
            valid_eos = {name for name in macroscopic_ops if name != "standard"}
            if self.eos not in valid_eos:
                raise ValueError(
                    f"eos must be one of {sorted(valid_eos)}, got '{self.eos}'"
                )
        else:
            valid_eos = {"double-well", "carnahan-starling"}
            if self.eos not in valid_eos:
                raise ValueError(f"eos must be one of {valid_eos}, got '{self.eos}'")

        # Validate collision_scheme using the registry (dynamically populated)
        collision_ops = get_operators("collision")
        if collision_ops:
            valid_schemes = {name for name, entry in collision_ops.items()
                            if name not in ("source_term",)}
            if self.collision_scheme not in valid_schemes:
                raise ValueError(
                    f"collision_scheme must be one of {sorted(valid_schemes)}, "
                    f"got '{self.collision_scheme}'"
                )
        else:
            valid_schemes = {"bgk", "mrt"}
            if self.collision_scheme not in valid_schemes:
                raise ValueError(f"collision_scheme must be one of {valid_schemes}, got '{self.collision_scheme}'")

        # Validate k_diag is provided for MRT
        if self.collision_scheme == "mrt" and self.k_diag is None:
            raise ValueError("k_diag must be provided when using MRT collision scheme")


@dataclass
class RunnerConfig:
    """
    Configuration for the simulation runner (I/O, saving, initialization).

    Attributes:
        save_interval: Save data every N timesteps.
        skip_interval: Skip saving for the first N timesteps (transient).
        results_dir: Base directory for saving results.
        init_type: Initialization type (e.g. "standard", "multiphase_droplet").
        init_dir: Path to .npz file for "init_from_file" init_type.
        simulation_name: Optional name for the simulation run.
        save_fields: List of field names to save (None = save all).
    """
    save_interval: int = 100
    skip_interval: int = 0
    results_dir: str = BASE_RESULTS_DIR
    init_type: str = "standard"
    init_dir: Optional[str] = None
    simulation_name: Optional[str] = None
    save_fields: Optional[List[str]] = None

    def __post_init__(self):
        # Validate save_interval
        if self.save_interval <= 0:
            raise ValueError(f"save_interval must be positive, got {self.save_interval}")

        # Validate skip_interval
        if self.skip_interval < 0:
            raise ValueError(f"skip_interval must be non-negative, got {self.skip_interval}")

        # Validate init_dir is provided for init_from_file
        if self.init_type == "init_from_file" and self.init_dir is None:
            raise ValueError("init_dir must be provided when init_type is 'init_from_file'")

        # Validate save_fields if provided
        if self.save_fields is not None:
            valid_fields = {"f", "rho", "u", "force", "force_ext", "h"}
            invalid = set(self.save_fields) - valid_fields
            if invalid:
                raise ValueError(f"Invalid save_fields: {invalid}. Valid fields: {valid_fields}")


# Type alias for simulation configs
SimulationConfig = Union[SinglePhaseConfig, MultiphaseConfig]


@dataclass
class SimulationBundle:
    """
    Top-level configuration object returned by file adapters.

    Bundles the simulation physics config and runner/IO config into one
    object that gets passed around and unpacked by subsystems that need
    specific parts.

    Example — Python API::

        from config import SimulationBundle, SinglePhaseConfig, RunnerConfig
        from core import Run

        bundle = SimulationBundle(
            simulation=SinglePhaseConfig(grid_shape=(100, 100), tau=0.6, nt=10000),
            runner=RunnerConfig(save_interval=1000),
        )
        sim = Run(bundle)
        sim.run()

    Example — multiphase::

        bundle = SimulationBundle(
            simulation=MultiphaseConfig(
                grid_shape=(401, 101), tau=0.99, kappa=0.017,
                rho_l=1.0, rho_v=0.33, interface_width=4,
                force_enabled=True, force_obj=[gravity],
                bc_config=bc_config,
            ),
            runner=RunnerConfig(save_interval=2000, init_type="wetting"),
        )

    Attributes:
        simulation: Physics configuration (SinglePhaseConfig or MultiphaseConfig).
        runner: Runner/IO configuration.
    """

    simulation: SimulationConfig
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    @property
    def is_multiphase(self) -> bool:
        """Whether this is a multiphase simulation."""
        return isinstance(self.simulation, MultiphaseConfig)

    @property
    def is_single_phase(self) -> bool:
        """Whether this is a single-phase simulation."""
        return isinstance(self.simulation, SinglePhaseConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire bundle to a flat dictionary.

        Merges simulation and runner configs into a single dict,
        adding a ``simulation_type`` key for backward-compatibility.
        """
        sim_dict = asdict(self.simulation)
        runner_dict = asdict(self.runner)

        # Merge extra into main dict
        extra = sim_dict.pop("extra", {})
        sim_dict.update(extra)

        # Add simulation_type for backward-compatibility
        if self.is_multiphase:
            sim_dict["simulation_type"] = "multiphase"
        else:
            sim_dict["simulation_type"] = "single_phase"

        # Runner keys go into the same flat dict
        sim_dict.update(runner_dict)
        return sim_dict

    def __repr__(self) -> str:
        sim_type = "multiphase" if self.is_multiphase else "single_phase"
        return (
            f"SimulationBundle(\n"
            f"  type={sim_type},\n"
            f"  simulation={self.simulation!r},\n"
            f"  runner={self.runner!r},\n"
            f")"
        )

