"""``[initialisation]``: the phase dispersed in the run, read once for every consumer.

A droplet (liquid dispersed in vapour) and a bubble (vapour dispersed in liquid)
share every multiphase operator, but a few setup-time choices depend on which one
the run is: the knob that makes a chemical step's destination surface wet the
dispersed phase more (``config.wetting.chemical_step_wall``) is one.
:func:`dispersed_phase` is the single place that answers the question.
"""

from __future__ import annotations
from enum import StrEnum
from typing import TYPE_CHECKING
from src.config.defaults import DISPERSED_PHASE

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig


class DispersedPhase(StrEnum):
    """The phase of the inclusion: ``[initialisation] dispersed``."""

    VAPOUR = "vapour"
    LIQUID = "liquid"


def dispersed_phase(config: SimulationConfig) -> DispersedPhase | None:
    """The phase dispersed in *config*'s run, or ``None`` when the config does not say.

    ``[initialisation] dispersed`` when set. Without it, a ``multiphase_bubbles``
    run uses the initialiser's default (:data:`~src.config.defaults.DISPERSED_PHASE`);
    any other initialiser (``init_from_file`` among them) leaves it unknown.
    """
    value = config.initialisation.get("dispersed")
    if value is not None:
        return DispersedPhase(value)
    if config.init_type == "multiphase_bubbles":
        return DispersedPhase(DISPERSED_PHASE)
    return None
