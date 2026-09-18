"""Initialisation from file — pure function (non-jittable).

Loads density and velocity fields from a NumPy ``.npz`` archive and
computes the equilibrium distribution.  This function involves file I/O
and is therefore only called at setup time, outside JIT.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from src.operators.equilibrium import build_equilibrium_fn
from src.registry import initialise_operator

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice


@initialise_operator(name="init_from_file")
def init_from_file(
    nx: int,
    ny: int,
    nz: int,
    lattice: Lattice,
    *,
    npz_path: str,
    **_kwargs: object,
) -> jnp.ndarray:
    """Load ``rho`` and ``u`` from an ``.npz`` file and compute equilibrium.

    The archive must contain arrays ``rho`` of shape ``(nx, ny, nz, 1, 1)``
    and ``u`` of shape ``(nx, ny, nz, 1, 2)``.

    Args:
        nx: Expected grid size in x.
        ny: Expected grid size in y.
        nz: Expected grid size in z.
        lattice: :class:`~setup.lattice.Lattice`.
        npz_path: Filesystem path to the ``.npz`` archive.
        **kwargs: Additional arguments (ignored).

    Returns:
        Initial distribution ``f``, shape ``(nx, ny, nz, q, 1)``.

    Raises:
        FileNotFoundError: If *npz_path* does not exist.
        ValueError: If the loaded shapes do not match ``(nx, ny, nz, ...)``.
    """
    equilibrium_fn = build_equilibrium_fn("wb")
    data = np.load(npz_path)
    rho = jnp.array(data["rho"])
    u = jnp.array(data["u"])
    if rho.shape != (nx, ny, nz, 1, 1):
        msg = f"Expected rho shape ({nx}, {ny}, {nz}, 1, 1), got {rho.shape}"
        raise ValueError(msg)
    if u.shape != (nx, ny, nz, 1, 2):
        msg = f"Expected u shape ({nx}, {ny}, {nz}, 1, 2), got {u.shape}"
        raise ValueError(msg)
    return equilibrium_fn(rho, u, lattice)
