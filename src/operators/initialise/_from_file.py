"""Initialisation from file — pure function (non-jittable).

Loads density and velocity fields from a NumPy ``.npz`` archive and
computes the equilibrium distribution.  This function involves file I/O
and is therefore only called at setup time, outside JIT.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from operators.equilibrium import build_equilibrium_fn
from registry import initialise_operator
from setup.lattice import Lattice


@initialise_operator(name="init_from_file")
def init_from_file(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    npz_path: str,
    **kwargs,
) -> jnp.ndarray:
    """Load ``rho`` and ``u`` from an ``.npz`` file and compute equilibrium.

    The archive must contain arrays ``rho`` of shape ``(nx, ny, 1, 1)``
    and ``u`` of shape ``(nx, ny, 1, 2)``.

    Args:
        nx: Expected grid size in x.
        ny: Expected grid size in y.
        lattice: :class:`~setup.lattice.Lattice`.
        npz_path: Filesystem path to the ``.npz`` archive.

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, q, 1)``.

    Raises:
        FileNotFoundError: If *npz_path* does not exist.
        ValueError: If the loaded shapes do not match ``(nx, ny, ...)``.
    """
    equilibrium_fn = build_equilibrium_fn("wb")
    data = np.load(npz_path)
    rho = jnp.array(data["rho"])
    u = jnp.array(data["u"])
    if rho.shape != (nx, ny, 1, 1):
        raise ValueError(f"Expected rho shape ({nx}, {ny}, 1, 1), got {rho.shape}")
    if u.shape != (nx, ny, 1, 2):
        raise ValueError(f"Expected u shape ({nx}, {ny}, 1, 2), got {u.shape}")
    return equilibrium_fn(rho, u, lattice)
