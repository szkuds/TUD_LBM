"""
StepResult dataclass for standardized simulation step output.

This dataclass provides a consistent interface for returning
macroscopic fields from simulation timesteps, avoiding re-computation
during data saving.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp


@dataclass
class StepResult:
    """
    Standardized result from a simulation timestep.

    Attributes:
        f: Distribution function array.
        rho: Density field (optional).
        u: Velocity field (optional).
        force: Force field from collision/interactions (optional).
        force_ext: External force field (optional).
        h: Electric potential field (optional).
    """
    f: jnp.ndarray
    rho: Optional[jnp.ndarray] = None
    u: Optional[jnp.ndarray] = None
    force: Optional[jnp.ndarray] = None
    force_ext: Optional[jnp.ndarray] = None
    h: Optional[jnp.ndarray] = None
