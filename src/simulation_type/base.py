from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from simulation_domain.lattice import Lattice
from simulation_domain.grid import Grid
from app_setup.registry import get_operators

if TYPE_CHECKING:
    from runner.step_result import StepResult
    from app_setup.simulation_setup import SimulationSetup


class BaseSimulation(ABC):
    def __init__(self, grid_shape, lattice_type="D2Q9", tau=1.0, nt=1000):
        self.grid_shape = grid_shape
        self.nt = nt
        self.grid = Grid(grid_shape)
        self.lattice = Lattice(lattice_type)
        self.tau = tau

        # Add simulation_type type flags
        self.multiphase = False
        self.wetting_enabled = False

    # ── Shared helpers for registry-based operator construction ────────

    @staticmethod
    def _make_initialiser(
        init_type: str,
        config: "SimulationSetup",
    ):
        """Resolve an initialisation operator from the registry and build it.

        Args:
            init_type: Registered name, e.g. ``"standard"``, ``"multiphase_droplet"``.
            config: Simulation configuration.

        Returns:
            An initialised :class:`InitialisationBase` subclass instance.

        Raises:
            KeyError: If *init_type* is not registered.
        """
        init_ops = get_operators("initialise")
        entry = init_ops.get(init_type)
        if entry is None:
            available = sorted(init_ops.keys())
            raise KeyError(
                f"Unknown initialisation type '{init_type}'. "
                f"Must be one of {available}."
            )
        return entry.cls.from_config(config)

    @staticmethod
    def _make_boundary_condition(
        config: "SimulationSetup",
    ):
        """Resolve the composite boundary condition from the registry and build it.

        Uses the ``"standard"`` (composite) BC, which internally inspects
        ``config.bc_config`` and dispatches per-edge to specialised operators.

        Args:
            config: Simulation configuration (``bc_config`` is guaranteed
                    non-None by ``__post_init__``).

        Returns:
            A :class:`BoundaryCondition` instance.
        """
        bc_ops = get_operators("boundary_condition")
        bc_cls = bc_ops["standard"].cls
        return bc_cls(config)

    # ── Abstract interface ────────────────────────────────────────────

    @abstractmethod
    def setup_operators(self):
        """Setup simulation_type-specific simulation_operators"""
        pass

    @abstractmethod
    def initialise_fields(self, init_type="standard", *, init_dir=None):
        """
        Parameters
        ----------
        init_type : str
            Name of the initialisation routine.
        init_dir : str or None, optional
            Path to the .npz snapshot when `init_type=="init_from_file"`.
        """
        pass

    @abstractmethod
    def run_timestep(self, f_prev, it, **kwargs) -> "StepResult":
        """Execute one timestep and return a StepResult with macroscopic fields."""
        pass
