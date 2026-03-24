"""Transient configuration for the differential-operators factory.

:class:`DifferentialConfig` carries everything
:func:`~operators.differential.factory.build_differential_operators` needs.
It is created inside :func:`~setup.simulation_setup.build_setup` and
discarded after the factory call — it is **never** stored on
:class:`~setup.simulation_setup.SimulationSetup`.
"""

from __future__ import annotations
from typing import Any
from typing import NamedTuple
import jax.numpy as jnp


class DifferentialConfig(NamedTuple):
    """All inputs needed to build gradient / Laplacian operators.

    Attributes:
        w:              Lattice weights, shape ``(q,)``.
        c:              Lattice velocity vectors, shape ``(2, q)``.
        pad_modes:      Four padding-mode strings
                        ``[right_y, left_y, bottom_x, top_x]``.
        wetting_params: ``None`` when wetting is disabled; otherwise the
                        dict accepted by :func:`make_wetting_gradient`.
        chemical_step:  Optional step index for chemical-step wetting
                        geometries.
        bc_config:      Boundary-condition config dict, e.g.
                        ``{"bottom": "wetting", "top": "bounce-back"}``.
                        Passed through to :func:`apply_wetting_to_all_edges`.
    """

    w: jnp.ndarray
    c: jnp.ndarray
    pad_modes: list[str]
    wetting_params: dict[str, Any] | None = None
    chemical_step: int | None = None
    bc_config: dict[str, Any] | None = None

    @property
    def wetting_enabled(self) -> bool:
        """Return ``True`` when wetting parameters are present."""
        return self.wetting_params is not None

    @property
    def chemical_step_enabled(self) -> bool:
        """Return ``True`` when chemical step parameters are present."""
        return self.chemical_step is not None
