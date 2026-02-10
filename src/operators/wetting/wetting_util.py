from typing import NamedTuple

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class WettingParameters(NamedTuple):
    d_rho_left: jnp.ndarray
    d_rho_right: jnp.ndarray
    phi_left: jnp.ndarray
    phi_right: jnp.ndarray

    def tree_flatten(self):
        return (self.d_rho_left, self.d_rho_right, self.phi_left, self.phi_right), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def determine_padding_modes(bc_config):
    if not bc_config:
        return ['wrap', 'wrap', 'wrap', 'wrap']
    padmode = ['wrap', 'wrap', 'wrap', 'wrap']
    for edge, bc_type in bc_config.items():
        if bc_type in ['symmetry', 'bounce-back', 'wetting']:
            if edge == 'bottom':
                padmode[0] = 'edge'
            elif edge == 'right':
                padmode[1] = 'edge'
            elif edge == 'top':
                padmode[2] = 'edge'
            elif edge == 'left':
                padmode[3] = 'edge'
    return padmode


def has_wetting_bc(bc_config):
    if not bc_config:
        return False
    return any(bc == 'wetting' for key, bc in bc_config.items()
               if key != 'wetting_params' and isinstance(bc, str))


def apply_wetting_to_all_edges(obj, grid_padded, rho_l, rho_v, phi_left, phi_right, d_rho_left, d_rho_right, width):
    for edge in ['bottom', 'top', 'left', 'right']:
        if obj.bc_config.get(edge) == 'wetting':
            if edge == 'bottom':  # y=0 border
                grid_padded = wetting_1d(grid_padded, axis=1, idx=0,
                                         rho_l=rho_l, rho_v=rho_v, phi_left=phi_left, phi_right=phi_right,
                                         d_rho_left=d_rho_left, d_rho_right=d_rho_right, width=width)
            elif edge == 'top':  # y=-1 border
                grid_padded = wetting_1d(grid_padded, axis=1, idx=-1,
                                         rho_l=rho_l, rho_v=rho_v, phi_left=phi_left, phi_right=phi_right,
                                         d_rho_left=d_rho_left, d_rho_right=d_rho_right, width=width)
            elif edge == 'left':  # x=0 border
                grid_padded = wetting_1d(grid_padded, axis=0, idx=0,
                                         rho_l=rho_l, rho_v=rho_v, phi_left=phi_left, phi_right=phi_right,
                                         d_rho_left=d_rho_left, d_rho_right=d_rho_right, width=width)
            elif edge == 'right':  # x=-1 border
                grid_padded = wetting_1d(grid_padded, axis=0, idx=-1,
                                         rho_l=rho_l, rho_v=rho_v, phi_left=phi_left, phi_right=phi_right,
                                         d_rho_left=d_rho_left, d_rho_right=d_rho_right, width=width)
    return grid_padded


def wetting_1d(arr, axis, idx, rho_l, rho_v, phi_left, phi_right, d_rho_left, d_rho_right, width):
    # axis == 1 for the y-edges
    if axis == 1:
        arr = arr.at[1:-1, idx].set(
            (1 / 3 * arr[1:-1, idx + 1 if idx == 0 else idx - 1] +
             1 / 12 * arr[0:-2, idx + 1 if idx == 0 else idx - 1] +
             1 / 12 * arr[2:, idx + 1 if idx == 0 else idx - 1])
            / (1 / 3 + 1 / 12 + 1 / 12))
        # Corners
        arr = arr.at[0, idx].set(
            (1 / 3 * arr[0, idx + 1 if idx == 0 else idx - 1] +
             1 / 12 * arr[-1, idx + 1 if idx == 0 else idx - 1] +
             1 / 12 * arr[1, idx + 1 if idx == 0 else idx - 1])
            / (1 / 3 + 1 / 12 + 1 / 12)
        )
        arr = arr.at[-1, idx].set(
            (1 / 3 * arr[-1, idx + 1 if idx == 0 else idx - 1] +
             1 / 12 * arr[0, idx + 1 if idx == 0 else idx - 1] +
             1 / 12 * arr[-2, idx + 1 if idx == 0 else idx - 1])
            / (1 / 3 + 1 / 12 + 1 / 12)
        )
        edge_slice = arr[1:-1, idx]

        mask1 = arr[1:-1, idx] < (0.95 * rho_l + 0.05 * rho_v)
        mask2 = arr[1:-1, idx] > (0.95 * rho_v + 0.05 * rho_l)
    # axis == 0 for the x-edges
    else:
        arr = arr.at[idx, -1:1].set(
            (1 / 3 * arr[idx + 1 if idx == 0 else idx - 1, 1:-1] +
             1 / 12 * arr[idx + 1 if idx == 0 else idx - 1, 0:-2] +
             1 / 12 * arr[idx + 1 if idx == 0 else idx - 1, 2:])
            / (1 / 3 + 1 / 12 + 1 / 12))
        #Corners
        arr = arr.at[idx, 0].set(
            (1 / 3 * arr[idx + 1 if idx == 0 else idx - 1, 0] +
             1 / 12 * arr[idx + 1 if idx == 0 else idx -1, -1] +
             1 / 12 * arr[idx + 1 if idx == 0 else idx -1, 1]) / (1 / 3 + 1 / 12 + 1 / 12)
        )
        arr = arr.at[idx, -1].set(
            (1 / 3 * arr[idx + 1 if idx == 0 else idx - 1, -1] +
             1 / 12 * arr[idx + 1 if idx == 0 else idx -1, 0] +
             1 / 12 * arr[idx + 1 if idx == 0 else idx -1, -2]) / (1 / 3 + 1 / 12 + 1 / 12)
        )
        edge_slice = arr[idx, 1:-1]

        mask1 = arr[idx, -1:1] < (0.95 * rho_l + 0.05 * rho_v)
        mask2 = arr[idx, -1:1] > (0.95 * rho_v + 0.05 * rho_l)

    # Wetting mask logic
    mask_final = mask1 * mask2

    mask1_int = jnp.array(mask1, dtype=int)
    diff_mask1 = jnp.diff(mask1_int)

    # Determining the transition index, the [0] is used to extract only the first value
    transition_index_left_mask1 = jnp.where(diff_mask1 == -1, size=1, fill_value=0)[0] + width
    transition_index_right_mask1 = (jnp.where(diff_mask1 == 1, size=1, fill_value=0)[0]) - width

    # Here the mask_final is split into a left and a right mask to enable the CA of the left and right side to be
    # determined separately, the reason left uses the right mask is that it works as a cover.
    indices = jnp.arange(mask_final.shape[0])
    mask_cover_left = jnp.where(indices >= transition_index_right_mask1[0], False, mask_final)
    mask_cover_right = jnp.where(indices <= transition_index_left_mask1[0], False, mask_final)

    new_values_left = jnp.minimum(
        jnp.maximum(
            ((phi_left * edge_slice) - d_rho_left),
            (0.95 * rho_v + 0.05 * rho_l)
        ),
        (0.95 * rho_l + 0.05 * rho_v)
    )

    new_values_right = jnp.minimum(
        jnp.maximum(
            ((phi_right * edge_slice) - d_rho_right),
            (0.95 * rho_v + 0.05 * rho_l)
        ),
        (0.95 * rho_l + 0.05 * rho_v)
    )
    updated_slice = jnp.where(mask_cover_right, new_values_right, edge_slice)
    updated_slice = jnp.where(mask_cover_left, new_values_left, updated_slice)


    if axis == 1:
        arr = arr.at[1:-1, idx].set(updated_slice)
    else:
        arr = arr.at[idx, 1:-1].set(updated_slice)
    return arr
