import jax.numpy as jnp
import numpy as np
import os
from src.core.lattice.lattice import Lattice
from src.core.grid.grid import Grid
from src.operators.equilibrium import Equilibrium


class Initialise:
    """
    Handles the initialisation of the simulation for various scenarios.
    """

    def __init__(self, grid: Grid, lattice: Lattice, bubble: bool = False, g: float = None, rho_ref: float = None):
        self.grid = grid
        self.lattice = lattice
        self.nx, self.ny = grid.nx, grid.ny
        self.q = self.lattice.q
        self.equilibrium = Equilibrium(self.grid, self.lattice)

    def initialise_standard(
            self, density: float = 1.0, velocity: np.ndarray = np.array([0.0, 0.0])
    ):
        """
        Initialises a standard simulation with uniform density and velocity.

        Args:
            density (float): Initial uniform density.
            velocity (np.ndarray): Initial uniform velocity [ux, uy].

        Returns:
            jnp.ndarray: Initialised population distribution f.
        """
        # Create density and velocity fields with the correct 4D shape
        rho = jnp.full((self.nx, self.ny, 1, 1), density)

        u = jnp.broadcast_to(
            jnp.array(velocity).reshape(1, 1, 1, 2), (self.nx, self.ny, 1, 2)
        )

        # Return the equilibrium distribution for this state
        return self.equilibrium(rho, u)

    def initialise_multiphase_droplet_top(
            self, rho_l: float, rho_v: float, interface_width: int
    ):
        """
        Initialises a multiphase simulation with a low-density bubble in the center.

        Args:
            rho_l (float): Liquid phase density.
            rho_v (float): Vapour phase (bubble) density.

        Returns:
            jnp.ndarray: Initialised population distribution f.
        """
        # Create a density field with a bubble in the center
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny), indexing="ij")
        center_x, center_y = self.nx // 2, 5 * self.ny // 6
        radius = min(self.nx, self.ny) // 4

        # Use tanh for a smooth, stable interface
        distance = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        rho_field_2d = (rho_l + rho_v) / 2 - (rho_l - rho_v) / 2 * jnp.tanh(
            (distance - radius) / interface_width
        )

        # Reshape to 4D
        rho = rho_field_2d.reshape((self.nx, self.ny, 1, 1))

        # Initialise with zero velocity
        u = jnp.zeros((self.nx, self.ny, 1, 2))

        # Return the equilibrium distribution
        return self.equilibrium(rho, u)

    def initialise_multiphase_bubble(
            self, rho_l: float, rho_v: float, interface_width: int
    ):
        """
        Initialises a multiphase simulation with a low-density bubble in the center.

        Args:
            rho_l (float): Liquid phase density.
            rho_v (float): Vapour phase (bubble) density.

        Returns:
            jnp.ndarray: Initialised population distribution f.
        """
        # Create a density field with a bubble in the center
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny), indexing="ij")
        center_x, center_y = self.nx // 2, self.ny // 2
        radius = min(self.nx, self.ny) // 4

        # Use tanh for a smooth, stable interface
        distance = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        rho_field_2d = (rho_l + rho_v) / 2 + (rho_l - rho_v) / 2 * jnp.tanh(
            (distance - radius) / interface_width
        )

        # Reshape to 4D
        rho = rho_field_2d.reshape((self.nx, self.ny, 1, 1))

        # Initialise with zero velocity
        u = jnp.zeros((self.nx, self.ny, 1, 2))

        # Return the equilibrium distribution
        return self.equilibrium(rho, u)

    def initialise_multiphase_bubble_bubble(
            self, rho_l: float, rho_v: float, interface_width: int
    ):
        """
        Initialises a multiphase simulation with two low-density bubbles
        Args:
            rho_l (float): Liquid phase density.
            rho_v (float): Vapour phase (bubble) density.
        returns
            jnp.ndarray: Initialised population distribution f.
        """
        #create a density field with two bubbles placed side-by-side
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny))
        left_bubble_center_x, left_bubble_center_y = self.nx // 4, self.ny // 2
        right_bubble_center_x, right_bubble_center_y = self.nx * 2.4 // 4, self.ny // 2
        radius = min(self.nx, self.ny) // 5

        #use tanh for a smooth, stable interface
        distance_to_left_bubble = jnp.sqrt((x - left_bubble_center_x) ** 2 + (y - left_bubble_center_y) ** 2)
        distance_to_right_bubble = jnp.sqrt((x - right_bubble_center_x) ** 2 + (y - right_bubble_center_y) ** 2)
        minimum_distance = jnp.minimum(distance_to_left_bubble, distance_to_right_bubble * 1.5)
        rho_field_2d = (rho_l + rho_v) / 2 + (rho_l - rho_v) / 2 * jnp.tanh(
            (minimum_distance - radius) / interface_width
        )
        # Reshape to 4D
        rho = rho_field_2d.reshape((self.nx, self.ny, 1, 1))

        #initialize with zero velocity
        u = jnp.zeros((self.nx, self.ny, 1, 2))

        #Return the f_eq
        return self.equilibrium(rho, u)

    def initialise_multiphase_lateral_bubble_configuration(
            self, rho_l: float, rho_v: float, interface_width: int
    ):
        """
        Initialises a multiphase simulation with two low-density bubbles
        Args:
            rho_l (float): Liquid phase density.
            rho_v (float): Vapour phase (bubble) density.
        returns
            jnp.ndarray: Initialised population distribution f.
        """
        #create a density field with two bubbles placed side-by-side
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny))
        left_bubble_center_x, left_bubble_center_y = self.nx // 2, self.ny * 2 // 6
        right_bubble_center_x, right_bubble_center_y = self.nx // 2, self.ny * 4 // 6
        radius = min(self.nx, self.ny) // 6.5

        #use tanh for a smooth, stable interface
        distance_to_left_bubble = jnp.sqrt((x - left_bubble_center_x) ** 2 + (y - left_bubble_center_y) ** 2)
        distance_to_right_bubble = jnp.sqrt((x - right_bubble_center_x) ** 2 + (y - right_bubble_center_y) ** 2)
        minimum_distance = jnp.minimum(distance_to_left_bubble, distance_to_right_bubble)
        rho_field_2d = (rho_l + rho_v) / 2 + (rho_l - rho_v) / 2 * jnp.tanh(
            (minimum_distance - radius) / interface_width
        )
        # Reshape to 4D
        rho = rho_field_2d.reshape((self.nx, self.ny, 1, 1))

        #initialize with zero velocity
        u = jnp.zeros((self.nx, self.ny, 1, 2))

        #Return the f_eq
        return self.equilibrium(rho, u)

    def initialise_multiphase_droplet(
            self, rho_l: float, rho_v: float, interface_width: int
    ):
        """
        Initialises a multiphase simulation with a droplet in the center.

        Args:
            rho_l (float): Liquid phase density.
            rho_v (float): Vapour phase (bubble) density.

        Returns:
            jnp.ndarray: Initialised population distribution f.
        """
        # Create a density field with a bubble in the center
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny), indexing="ij")
        center_x, center_y = self.nx // 2, self.ny // 2
        radius = min(self.nx, self.ny) // 8

        # Use tanh for a smooth, stable interface
        distance = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        rho_field_2d = (rho_l + rho_v) / 2 - (rho_l - rho_v) / 2 * jnp.tanh(
            (distance - radius) / interface_width
        )

        # Reshape to 4D
        rho = rho_field_2d.reshape((self.nx, self.ny, 1, 1))

        # Initialise with zero velocity
        u = jnp.zeros((self.nx, self.ny, 1, 2))

        # Return the equilibrium distribution
        return self.equilibrium(rho, u)

    def initialise_multiphase_bubble_bot(
            self, rho_l: float, rho_v: float, interface_width: int
    ):
        """
        Initialises a multiphase simulation with a low-density bubble in the center.

        Args:
            rho_l (float): Liquid phase density.
            rho_v (float): Vapour phase (bubble) density.

        Returns:
            jnp.ndarray: Initialised population distribution f.
        """
        # Create a density field with a bubble in the center
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny), indexing="ij")
        center_x, center_y = self.nx // 2, self.ny // 6
        radius = min(self.nx, self.ny) // 4

        # Use tanh for a smooth, stable interface
        distance = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        rho_field_2d = (rho_l + rho_v) / 2 + (rho_l - rho_v) / 2 * jnp.tanh(
            (distance - radius) / interface_width
        )

        # Reshape to 4D
        rho = rho_field_2d.reshape((self.nx, self.ny, 1, 1))

        # Initialise with zero velocity
        u = jnp.zeros((self.nx, self.ny, 1, 2))

        # Return the equilibrium distribution
        return self.equilibrium(rho, u)

    def initialise_wetting_chemical_step(
            self, rho_l: float, rho_v: float, interface_width: int
    ):
        """
        Initialize the simulation with a droplet wetting a solid surface.

        Args:
            rho_l (float): Liquid phase density.
            rho_v (float): Vapour phase density.
            interface_width (int): Width of the interface for tanh profile.

        Returns:
            jnp.ndarray: Initial distribution function.
        """
        # Radius of the droplet (adapted from user query)
        r = (self.ny) / 3.3

        # Initialize velocity (zero) and density fields with correct shapes
        u = jnp.zeros((self.nx, self.ny, 1, 2))
        rho = jnp.zeros((self.nx, self.ny, 1, 1))

        # Create grid (shifted by 0.5 for cell centers)
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny), indexing="ij")

        # Calculate center coordinates (droplet centered horizontally, near bottom)
        xc, yc = self.nx / 2, self.ny / 2

        # Calculate distance from center (shifted to simulate wetting at bottom)
        distance = jnp.sqrt((x - xc / 2) ** 2 + (y) ** 2)

        # Calculate density distribution using tanh for smooth interface
        rho_2d = (rho_l + rho_v) / 2 + (rho_l - rho_v) / 2 * jnp.tanh(
            2 * (r - distance) / interface_width
        )

        # Assign to rho (reshape to 4D)
        rho = rho.at[:, :, 0, 0].set(rho_2d)

        # Return equilibrium distribution
        return self.equilibrium(rho, u)

    def initialise_wetting(self, rho_l: float, rho_v: float, interface_width: int):
        """
        Initialize the simulation with a droplet wetting a solid surface.

        Args:
            rho_l (float): Liquid phase density.
            rho_v (float): Vapour phase density.
            interface_width (int): Width of the interface for tanh profile.

        Returns:
            jnp.ndarray: Initial distribution function.
        """
        # Radius of the droplet (adapted from user query)
        r = (self.ny) / 3.33

        # Initialize velocity (zero) and density fields with correct shapes
        u = jnp.zeros((self.nx, self.ny, 1, 2))
        rho = jnp.zeros((self.nx, self.ny, 1, 1))

        # Create grid (shifted by 0.5 for cell centers)
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny), indexing="ij")

        # Calculate center coordinates (droplet centered horizontally, near bottom)
        xc, yc = self.nx / 2, self.ny / 2

        # Calculate distance from center (shifted to simulate wetting at bottom)
        distance = jnp.sqrt((x - xc) ** 2 + (y) ** 2)

        # Calculate density distribution using tanh for smooth interface
        rho_2d = (rho_l + rho_v) / 2 + (rho_l - rho_v) / 2 * jnp.tanh(
            2 * (r - distance) / interface_width
        )

        # Assign to rho (reshape to 4D)
        rho = rho.at[:, :, 0, 0].set(rho_2d)

        # Return equilibrium distribution
        return self.equilibrium(rho, u)

    def init_from_npz(self, npz_path: str):
        """
        Initialise the simulation from a saved state containing only macroscopic
        fields (rho, u) stored in a compressed NumPy ``.npz`` file.

        Parameters
        ----------
        npz_path : str
            Absolute or relative path to the ``.npz`` file.

        Returns
        -------
        jnp.ndarray
            The initialised 4-D distribution function ``f`` created using the
            equilibrium populations for the given rho and u.

        Raises
        ------
        AssertionError
            If the array dimensions do not match the current grid.
        FileNotFoundError
            If *npz_path* does not exist.
        ValueError
            If the file does not contain both 'rho' and 'u' keys.
        """
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Could not locate file: {npz_path}")
        data = np.load(npz_path)

        if not {"rho", "u"}.issubset(data.files):
            raise ValueError(
                "Missing required keys in restart file: both 'rho' and 'u' must be present. "
                f"Available keys: {list(data.files)}"
            )

        rho = data["rho"]
        u = data["u"]

        # Shape checks (customise if your grid or file dimensions differ)
        assert rho.shape == (
            self.nx,
            self.ny,
            1,
            1,
        ), f"rho shape mismatch – expected ({self.nx}, {self.ny}, 1, 1) but got {rho.shape}"
        assert u.shape == (
            self.nx,
            self.ny,
            1,
            2,
        ), f"u shape mismatch – expected ({self.nx}, {self.ny}, 1, 2) but got {u.shape}"

        rho_jax = jnp.array(rho)
        u_jax = jnp.array(u)
        return self.equilibrium(rho_jax, u_jax)
