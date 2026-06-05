"""Legacy Carnahan-Starling macroscopic wrapper.

This module is kept for backward compatibility with
``build_macroscopic_fn("carnahan-starling")`` callers.
The main multiphase path now uses the unified ``multiphase`` operator
with EOS dispatch handled in ``tud_lbm.operators.macroscopic.eos``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tud_lbm.operators.macroscopic._multiphase import compute_macroscopic_multiphase
from tud_lbm.registry import macroscopic_operator

if TYPE_CHECKING:
    import jax.numpy as jnp
    from tud_lbm.lattice.lattice import Lattice
    from tud_lbm.operators.macroscopic import MultiphaseParams
    from tud_lbm.operators.protocols import DifferentialOperator


@macroscopic_operator(name="carnahan-starling")
def compute_macroscopic_multiphase_cs(
    f: jnp.ndarray,
    lattice: Lattice,
    mp: MultiphaseParams,
    force_ext: jnp.ndarray | None = None,
    *,
    gradient_standard: DifferentialOperator,
    laplacian_density: DifferentialOperator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compatibility alias that forces Carnahan-Starling EOS selection."""
    mp_cs = mp._replace(eos="carnahan-starling")
    return compute_macroscopic_multiphase(
        f,
        lattice,
        mp_cs,
        force_ext=force_ext,
        gradient_standard=gradient_standard,
        laplacian_density=laplacian_density,
    )
