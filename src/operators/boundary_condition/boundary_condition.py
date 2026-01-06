from functools import partial
import jax.numpy as jnp
from jax import jit
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.grid.grid import Grid
    from src.core.lattice.lattice import Lattice


class BoundaryCondition:
    """
    Applies boundary conditions to the post-streaming distribution function.
    Supports bounce-back, symmetry, and periodic BCs on specified grid edges.
    Uses dynamic indices from Lattice class instead of hardcoding.
    """

    def __init__(self, grid: "Grid", lattice: "Lattice", bc_config: Dict[str, str]):
        self.grid = grid
        self.lattice = lattice
        self.bc_config = bc_config
        self.opp_indices = lattice.opp_indices
        self.edges = grid.get_edges()
        valid_edges = ['top', 'bottom', 'left', 'right']
        valid_types = ['bounce-back', 'symmetry', 'periodic', 'wetting']
        for edge, bc_type in bc_config.items():
            # Skip wetting_params as it is not an edge boundary condition
            if edge == 'wetting_params' or edge == 'hysteresis_params' or edge == 'chemical_step':
                continue

            if edge not in valid_edges:
                raise ValueError(f'Invalid edge: {edge}. Must be one of {valid_edges}.')
            if bc_type not in valid_types:
                raise ValueError(
                    f'Invalid BC type: {bc_type}. Must be one of {valid_types}.'
                )

    @partial(jit, static_argnums=(0,))
    def __call__(
        self, f_streamed: jnp.ndarray, f_collision: jnp.ndarray
    ) -> jnp.ndarray:
        for edge, bc_type in self.bc_config.items():
            if bc_type == 'bounce-back' or bc_type == 'wetting':
                f_streamed = self._apply_bounce_back(f_streamed, f_collision, edge)
            elif bc_type == 'symmetry':
                f_streamed = self._apply_symmetry(f_streamed, f_collision, edge)
            elif bc_type == 'periodic':
                f_streamed = self._apply_periodic(f_streamed)
        return f_streamed

    @partial(jit, static_argnums=(0, 3))
    def _apply_bounce_back(
        self, f_streamed: jnp.ndarray, f_collision: jnp.ndarray, edge: str
    ) -> jnp.ndarray:
        lattice = self.lattice
        if edge == 'bottom':
            idx = 0
            incoming_dirs = lattice.construct_top_indices
            for i in incoming_dirs:
                opp_i = self.opp_indices[i]
                f_streamed = f_streamed.at[:, idx, i, 0].set(
                    f_collision[:, idx, opp_i, 0]
                )
        elif edge == 'top':
            idx = -1
            incoming_dirs = lattice.construct_bottom_indices
            for i in incoming_dirs:
                opp_i = self.opp_indices[i]
                f_streamed = f_streamed.at[:, idx, i, 0].set(
                    f_collision[:, idx, opp_i, 0]
                )
        elif edge == 'left':
            idx = 0
            incoming_dirs = lattice.construct_right_indices
            for i in incoming_dirs:
                opp_i = self.opp_indices[i]
                f_streamed = f_streamed.at[idx, :, i, 0].set(
                    f_collision[idx, :, opp_i, 0]
                )
        elif edge == 'right':
            idx = -1
            incoming_dirs = lattice.construct_left_indices
            for i in incoming_dirs:
                opp_i = self.opp_indices[i]
                f_streamed = f_streamed.at[idx, :, i, 0].set(
                    f_collision[idx, :, opp_i, 0]
                )
        return f_streamed

    @partial(jit, static_argnums=(0, 3))
    def _apply_symmetry(
        self, f_streamed: jnp.ndarray, f_collision: jnp.ndarray, edge: str
    ) -> jnp.ndarray:
        lattice = self.lattice
        if edge == 'bottom':
            idx = 0
            top_dirs = lattice.construct_top_indices
            bottom_dirs = lattice.construct_bottom_indices
            diag_top_right = lattice.construct_top_indices[2]
            diag_bottom_right = lattice.construct_bottom_indices[2]
            diag_top_left = lattice.construct_top_indices[1]
            diag_bottom_left = lattice.construct_bottom_indices[1]
            f_streamed = f_streamed.at[:, idx, top_dirs[0], 0].set(
                f_collision[:, idx, bottom_dirs[0], 0]
            )
            f_streamed = f_streamed.at[:, idx, diag_top_right, 0].set(
                f_collision[:, idx, diag_bottom_right, 0]
            )
            f_streamed = f_streamed.at[:, idx, diag_top_left, 0].set(
                f_collision[:, idx, diag_bottom_left, 0]
            )
        elif edge == 'top':
            idx = -1
            bottom_dirs = lattice.construct_bottom_indices
            top_dirs = lattice.construct_top_indices
            f_streamed = f_streamed.at[:, idx, bottom_dirs[0], 0].set(
                f_collision[:, idx, top_dirs[0], 0]
            )
            f_streamed = f_streamed.at[:, idx, bottom_dirs[1], 0].set(
                jnp.roll(f_collision[:, idx, top_dirs[1], 0], 1, axis=0)
            )
            f_streamed = f_streamed.at[:, idx, bottom_dirs[2], 0].set(
                jnp.roll(f_collision[:, idx, top_dirs[2], 0], -1, axis=0)
            )
        elif edge == 'left':
            idx = 0
            right_dirs = lattice.construct_right_indices
            left_dirs = lattice.construct_left_indices
            f_streamed = f_streamed.at[idx, :, right_dirs[0], 0].set(
                f_collision[idx, :, left_dirs[0], 0]
            )
            f_streamed = f_streamed.at[idx, :, right_dirs[1], 0].set(
                f_collision[idx, :, left_dirs[1], 0]
            )
            f_streamed = f_streamed.at[idx, :, right_dirs[2], 0].set(
                f_collision[idx, :, left_dirs[2], 0]
            )
        elif edge == 'right':
            idx = -1
            left_dirs = lattice.construct_left_indices
            right_dirs = lattice.construct_right_indices
            f_streamed = f_streamed.at[idx, :, left_dirs[0], 0].set(
                f_collision[idx, :, right_dirs[0], 0]
            )
            f_streamed = f_streamed.at[idx, :, left_dirs[1], 0].set(
                f_collision[idx, :, right_dirs[1], 0]
            )
            f_streamed = f_streamed.at[idx, :, left_dirs[2], 0].set(
                f_collision[idx, :, right_dirs[2], 0]
            )
        return f_streamed

    @partial(jit, static_argnums=(0,))
    def _apply_periodic(self, f_streamed: jnp.ndarray) -> jnp.ndarray:
        # For periodic boundaries, no additional transformation is needed
        # The streaming step already handles periodicity correctly
        return f_streamed
