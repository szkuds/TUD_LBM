from functools import partial

import jax.numpy as jnp
from jax import jit

from src.core.lattice.lattice import Lattice
from src.operators.wetting import (
    determine_padding_modes,
    has_wetting_bc,
    apply_wetting_to_all_edges,
)


class Gradient:
    """
    Callable class to calculate the gradient of a field using the LBM stencil,
    supporting asymmetric per-side padding.

    The implementation of the gradient operator is based on https://doi.org/10.1063/5.0072221
    """

    def __init__(self, lattice: Lattice, bc_config: dict = None):
        self.w = lattice.w
        self.c = lattice.c
        self.bc_config = bc_config
        self.pad_mode = determine_padding_modes(bc_config)
        # Only extract wetting parameters if wetting BC is present
        self.wetting_params = None
        if self.bc_config and 'chemical_step' in self.bc_config and has_wetting_bc(bc_config):
            self.wetting_params = bc_config.get('wetting_params')
            if self.wetting_params is None:
                raise ValueError("Wetting boundary condition specified but 'wetting_params' not found in bc_config")
            self.chemical_step = bc_config.get('chemical_step')
        elif has_wetting_bc(bc_config):
            self.wetting_params = bc_config.get('wetting_params')
            if self.wetting_params is None:
                raise ValueError("Wetting boundary condition specified but 'wetting_params' not found in bc_config")

    @partial(jit, static_argnums=(0,))
    def __call__(self, grid, pad_mode: list = None):
        """
        Calculate the gradient using the provided stencil and per-side boundary modes.

        Args:
            grid (jnp.ndarray): Input field, shape (nx, ny, 1, 1)
            pad_mode (list, optional): List of padding modes for each pad step

        Returns:
            jnp.ndarray: Gradient field, shape (nx, ny, 1, 2)
        """
        if self.wetting_params is not None:  # Only use wetting if params are available
            return self.wetting(grid, pad_mode)
        else:
            return self.standard(grid, pad_mode)

    def standard(self, grid, pad_mode):
        """Standard gradient calculation."""
        # Use provided pad_mode or fall back to self.pad_mode
        effective_pad_mode = pad_mode if pad_mode is not None else self.pad_mode

        # Extract 2D data from 4D input
        if grid.ndim == 4:
            grid_2d = grid[:, :, 0, 0]  # Extract (nx, ny) from (nx, ny, 1, 1)
        else:
            grid_2d = grid

        w = self.w
        c = self.c

        grad_ = jnp.zeros((2, grid_2d.shape[0], grid_2d.shape[1]))

        # Apply asymmetric per-side padding (same convention/order as Laplacian)
        grid_padded___ = jnp.pad(grid_2d, pad_width=((0, 0), (0, 1)), mode=effective_pad_mode[0])
        grid_padded__ = jnp.pad(grid_padded___, pad_width=((0, 0), (1, 0)), mode=effective_pad_mode[1])
        grid_padded_ = jnp.pad(grid_padded__, pad_width=((0, 1), (0, 0)), mode=effective_pad_mode[2])
        grid_padded = jnp.pad(grid_padded_, pad_width=((1, 0), (0, 0)), mode=effective_pad_mode[3])

        # Side nodes
        grid_ineg1_j0 = grid_padded[:-2, 1:-1]
        grid_ipos1_j0 = grid_padded[2:, 1:-1]
        grid_i0_jneg1 = grid_padded[1:-1, :-2]
        grid_i0_jpos1 = grid_padded[1:-1, 2:]

        # Corner nodes
        grid_ipos1_jpos1 = grid_padded[2:, 2:]
        grid_ineg1_jpos1 = grid_padded[:-2, 2:]
        grid_ineg1_jneg1 = grid_padded[:-2, :-2]
        grid_ipos1_jneg1 = grid_padded[2:, :-2]

        grad_ = grad_.at[0, :, :].set(
            3
            * (
                    w[1] * c[0, 1] * grid_ipos1_j0
                    + w[3] * c[0, 3] * grid_ineg1_j0
                    + w[5] * c[0, 5] * grid_ipos1_jpos1
                    + w[6] * c[0, 6] * grid_ineg1_jpos1
                    + w[7] * c[0, 7] * grid_ineg1_jneg1
                    + w[8] * c[0, 8] * grid_ipos1_jneg1
            )
        )

        grad_ = grad_.at[1, :, :].set(
            3
            * (
                    w[2] * c[1, 2] * grid_i0_jpos1
                    + w[4] * c[1, 4] * grid_i0_jneg1
                    + w[5] * c[1, 5] * grid_ipos1_jpos1
                    + w[6] * c[1, 6] * grid_ineg1_jpos1
                    + w[7] * c[1, 7] * grid_ineg1_jneg1
                    + w[8] * c[1, 8] * grid_ipos1_jneg1
            )
        )

        # Convert to 4D format: (nx, ny, 1, 2)
        grad_4d = jnp.zeros((grid_2d.shape[0], grid_2d.shape[1], 1, 2))
        grad_4d = grad_4d.at[:, :, 0, 0].set(grad_[0, :, :])
        grad_4d = grad_4d.at[:, :, 0, 1].set(grad_[1, :, :])

        return grad_4d

    def wetting(self, grid, pad_mode):
        rho_l = self.wetting_params['rho_l']
        rho_v = self.wetting_params['rho_v']
        width = self.wetting_params['width']
        weights = self.w
        c = self.c

        if getattr(self, "chemical_step", False):
            phi_left = jnp.ones(grid.shape[0])
            phi_left = (phi_left.at[(grid.shape[0] // int(1 / self.chemical_step['chemical_step_location'])):]
                        .set(self.wetting_params['phi_left']))
            d_rho_left = jnp.zeros(grid.shape[0])
            d_rho_left = (d_rho_left.at[:(grid.shape[0] // int(1 / self.chemical_step['chemical_step_location']))]
                          .set(self.wetting_params['d_rho_left']))
            phi_right = jnp.ones(grid.shape[0])
            phi_right = (phi_right.at[(grid.shape[0] // int(1 / self.chemical_step['chemical_step_location'])):]
                         .set(self.wetting_params['phi_right']))
            d_rho_right = jnp.zeros(grid.shape[0])
            d_rho_right = (d_rho_right.at[:(grid.shape[0] // int(1 / self.chemical_step['chemical_step_location']))]
                           .set(self.wetting_params['d_rho_right']))
        else:
            phi_left = self.wetting_params['phi_left']
            phi_right = self.wetting_params['phi_right']
            d_rho_left = self.wetting_params['d_rho_left']
            d_rho_right = self.wetting_params['d_rho_right']

        effective_pad_mode = pad_mode if pad_mode is not None else self.pad_mode
        if grid.ndim == 4:
            grid2d = grid[:, :, 0, 0]
        else:
            grid2d = grid

        grid_padded = jnp.pad(grid2d, ((0, 0), (0, 1)), mode=effective_pad_mode[0])
        grid_padded = jnp.pad(grid_padded, ((0, 0), (1, 0)), mode=effective_pad_mode[1])
        grid_padded = jnp.pad(grid_padded, ((0, 1), (0, 0)), mode=effective_pad_mode[2])
        grid_padded = jnp.pad(grid_padded, ((1, 0), (0, 0)), mode=effective_pad_mode[3])

        # Apply wetting to any relevant edge
        grid_padded = apply_wetting_to_all_edges(
            self, grid_padded, rho_l, rho_v, phi_left, phi_right, d_rho_left, d_rho_right, width
        )

        # STANDARD GRADIENT STENCIL
        grid_ineg1_j0 = grid_padded[:-2, 1:-1]
        grid_ipos1_j0 = grid_padded[2:, 1:-1]
        grid_i0_jneg1 = grid_padded[1:-1, :-2]
        grid_i0_jpos1 = grid_padded[1:-1, 2:]
        grid_ipos1_jpos1 = grid_padded[2:, 2:]
        grid_ineg1_jpos1 = grid_padded[:-2, 2:]
        grid_ineg1_jneg1 = grid_padded[:-2, :-2]
        grid_ipos1_jneg1 = grid_padded[2:, :-2]

        grad0 = (3 * (
                weights[1] * c[0, 1] * grid_ipos1_j0 +
                weights[3] * c[0, 3] * grid_ineg1_j0 +
                weights[5] * c[0, 5] * grid_ipos1_jpos1 +
                weights[6] * c[0, 6] * grid_ineg1_jpos1 +
                weights[7] * c[0, 7] * grid_ineg1_jneg1 +
                weights[8] * c[0, 8] * grid_ipos1_jneg1
        ))

        grad1 = (3 * (
                weights[2] * c[1, 2] * grid_i0_jpos1 +
                weights[4] * c[1, 4] * grid_i0_jneg1 +
                weights[5] * c[1, 5] * grid_ipos1_jpos1 +
                weights[6] * c[1, 6] * grid_ineg1_jpos1 +
                weights[7] * c[1, 7] * grid_ineg1_jneg1 +
                weights[8] * c[1, 8] * grid_ipos1_jneg1
        ))

        grad4d = jnp.zeros(grad0.shape + (1, 2))
        grad4d = grad4d.at[..., 0, 0].set(grad0)
        grad4d = grad4d.at[..., 0, 1].set(grad1)
        return grad4d

