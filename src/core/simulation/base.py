from abc import ABC, abstractmethod
from src.core.lattice.lattice import Lattice
from src.core.grid.grid import Grid


class BaseSimulation(ABC):
    def __init__(self, grid_shape, lattice_type="D2Q9", tau=1.0, nt=1000):
        self.grid_shape = grid_shape
        self.nt = nt
        self.grid = Grid(grid_shape)
        self.lattice = Lattice(lattice_type)
        self.tau = tau

        # Add simulation type flags
        self.multiphase = False
        self.wetting_enabled = False

    @abstractmethod
    def setup_operators(self):
        """Setup simulation-specific operators"""
        pass

    @abstractmethod
    def initialize_fields(self, init_type="standard", *, init_dir=None):
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
    def run_timestep(self, fprev, it):
        """Execute one timestep"""
        pass
