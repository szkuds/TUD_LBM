import jax.numpy as jnp


class ContactLineLocation:
    """
    Class to calculate contact line locations (left and right) from density and angles.
    """

    def __init__(self, rho_mean):
        self.rho_mean = rho_mean

    def compute(self, rho_, ca_left, ca_right):
        rho = rho_[:,:,0,0]
        array_i_j0 = rho[:, 0]

        mask_i_j0 = array_i_j0 < self.rho_mean

        mask_int_i_j0 = jnp.array(mask_i_j0, dtype=int)

        diff_mask_i_j0 = jnp.diff(mask_int_i_j0)

        transition_index_left_i_j0 = jnp.where(
            diff_mask_i_j0 == -1, size=1, fill_value=0
        )[0]
        transition_index_right_i_j0 = (
            jnp.where(diff_mask_i_j0 == 1, size=1, fill_value=0)[0]
        ) + 1

        index_left_i_j0 = transition_index_left_i_j0[0].astype(int)
        index_right_i_j0 = transition_index_right_i_j0[0].astype(int)

        x_val_left_j0 = index_left_i_j0 + (
            self.rho_mean - array_i_j0[index_left_i_j0]
        ) / (array_i_j0[index_left_i_j0 + 1] - array_i_j0[index_left_i_j0])
        x_val_right_j0 = (index_right_i_j0) - (
            self.rho_mean - array_i_j0[index_right_i_j0]
        ) / (array_i_j0[index_right_i_j0 - 1] - array_i_j0[index_right_i_j0])

        x_val_left_jsolid = x_val_left_j0 - (1 / (2 * jnp.tan(jnp.deg2rad(ca_left))))
        x_val_right_jsolid = x_val_right_j0 + (1 / (2 * jnp.tan(jnp.deg2rad(ca_right))))

        return x_val_left_jsolid, x_val_right_jsolid
