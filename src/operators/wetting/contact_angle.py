import jax.numpy as jnp
import math


class ContactAngle:
    """
    Class to calculate contact angles (left and right) from a density field.
    """

    def __init__(self, rho_mean):
        self.rho_mean = rho_mean

    def compute(self, rho_):
        rho = rho_[:,:,0,0]
        array_i_j0 = rho[:, 1]
        array_i_jpos1 = rho[:, 2]

        mask_i_j0 = array_i_j0 < self.rho_mean
        mask_i_jpos1 = array_i_jpos1 < self.rho_mean

        mask_int_i_j0 = jnp.array(mask_i_j0, dtype=int)
        mask_int_i_jpos1 = jnp.array(mask_i_jpos1, dtype=int)

        diff_mask_i_j0 = jnp.diff(mask_int_i_j0)
        diff_mask_i_jpos1 = jnp.diff(mask_int_i_jpos1)

        transition_index_left_i_j0 = jnp.where(
            diff_mask_i_j0 == -1, size=1, fill_value=0
        )[0]
        transition_index_left_i_jpos1 = jnp.where(
            diff_mask_i_jpos1 == -1, size=1, fill_value=0
        )[0]
        transition_index_right_i_j0 = (
            jnp.where(diff_mask_i_j0 == 1, size=1, fill_value=0)[0]
        ) + 1
        transition_index_right_i_jpos1 = (
            jnp.where(diff_mask_i_jpos1 == 1, size=1, fill_value=0)[0] + 1
        )

        index_left_i_j0 = transition_index_left_i_j0[0].astype(int)
        index_left_i_jpos1 = transition_index_left_i_jpos1[0].astype(int)
        index_right_i_j0 = transition_index_right_i_j0[0].astype(int)
        index_right_i_jpos1 = transition_index_right_i_jpos1[0].astype(int)

        x_val_left_j0 = index_left_i_j0 + (
            self.rho_mean - array_i_j0[index_left_i_j0]
        ) / (array_i_j0[index_left_i_j0 + 1] - array_i_j0[index_left_i_j0])
        x_val_left_jpos1 = index_left_i_jpos1 + (
            self.rho_mean - array_i_jpos1[index_left_i_jpos1]
        ) / (array_i_jpos1[index_left_i_jpos1 + 1] - array_i_jpos1[index_left_i_jpos1])
        x_val_right_j0 = index_right_i_j0 - (
            self.rho_mean - array_i_j0[index_right_i_j0]
        ) / (array_i_j0[index_right_i_j0 - 1] - array_i_j0[index_right_i_j0])
        x_val_right_jpos1 = index_right_i_jpos1 - (
            self.rho_mean - array_i_jpos1[index_right_i_jpos1]
        ) / (
            array_i_jpos1[index_right_i_jpos1 - 1] - array_i_jpos1[index_right_i_jpos1]
        )

        contact_angle_left = jnp.rad2deg(
            math.pi / 2 + jnp.arctan(x_val_left_j0 - x_val_left_jpos1)
        )
        contact_angle_right = jnp.rad2deg(
            math.pi / 2 + jnp.arctan(x_val_right_jpos1 - x_val_right_j0)
        )

        return contact_angle_left, contact_angle_right
