"""Initialisation from a saved ``.npz`` file.

Loads macroscopic fields (``rho``, ``u``) from a compressed NumPy file
and reconstructs the equilibrium distribution function.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from app_setup.registry import register_operator
from .base import InitialisationBase


@register_operator("initialise")
@dataclass
class InitialiseFromFile(InitialisationBase):
    """Restart from a saved ``.npz`` file.

    Registered as ``"init_from_file"``.
    """

    name = "init_from_file"

    def __call__(self, npz_path: str = "", **kwargs) -> jnp.ndarray:
        """Load ``rho`` and ``u`` from *npz_path* and return f_eq.

        Args:
            npz_path: Path to a ``.npz`` file containing ``rho`` and ``u``.

        Returns:
            4-D JAX array ``f``.

        Raises:
            FileNotFoundError: If *npz_path* does not exist.
            ValueError: If the file lacks ``rho`` or ``u`` keys.
            AssertionError: If array shapes don't match the grid.
        """
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Could not locate file: {npz_path}")

        data = np.load(npz_path)

        if not {"rho", "u"}.issubset(data.files):
            raise ValueError(
                "Missing required keys in restart file: both 'rho' and 'u' "
                f"must be present. Available keys: {list(data.files)}"
            )

        rho = data["rho"]
        u = data["u"]

        assert rho.shape == (
            self.nx, self.ny, 1, 1,
        ), (
            f"rho shape mismatch – expected ({self.nx}, {self.ny}, 1, 1) "
            f"but got {rho.shape}"
        )
        assert u.shape == (
            self.nx, self.ny, 1, 2,
        ), (
            f"u shape mismatch – expected ({self.nx}, {self.ny}, 1, 2) "
            f"but got {u.shape}"
        )

        rho_jax = jnp.array(rho)
        u_jax = jnp.array(u)
        return self.equilibrium(rho_jax, u_jax)

