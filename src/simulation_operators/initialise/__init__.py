"""Field initialisation routines for LBM simulations.

Provides initialisation of distribution functions and macroscopic fields
for various initial conditions and restart scenarios.

All concrete initialisation classes self-register in the global operator
registry via ``@register_operator("initialise")``.  Adding a new
initialisation type only requires creating a decorated subclass in a new
file and importing it here.

Classes:
    InitialisationBase: Abstract base class for initialisations.
    StandardInitialisation: Uniform density/velocity (``"standard"``).
    InitialiseMultiphaseBubble: Bubble in the centre (``"multiphase_bubble"``).
    InitialiseMultiphaseBubbleBot: Bubble near bottom (``"multiphase_bubble_bot"``).
    InitialiseMultiphaseBubbleBubble: Two side-by-side bubbles (``"multiphase_bubble_bubble"``).
    InitialiseMultiphaseDroplet: Droplet in the centre (``"multiphase_droplet"``).
    InitialiseMultiphaseDropletTop: Droplet near top (``"multiphase_droplet_top"``).
    InitialiseMultiphaseLateralBubble: Two stacked bubbles (``"multiphase_lateral_bubble_configuration"``).
    InitialiseMultiphaseDropletVariableRadius: Droplet with configurable radius (``"multiphase_droplet_variable_radius"``).
    InitialiseWetting: Wetting droplet at bottom wall (``"wetting"``).
    InitialiseWettingChemicalStep: Wetting with chemical step (``"wetting_chem_step"``).
    InitialiseFromFile: Restart from ``.npz`` file (``"init_from_file"``).
"""

from .base import InitialisationBase
from .standard import StandardInitialisation
from .multiphase_bubble import InitialiseMultiphaseBubble
from .multiphase_bubble_bot import InitialiseMultiphaseBubbleBot
from .multiphase_bubble_bubble import InitialiseMultiphaseBubbleBubble
from .multiphase_droplet import InitialiseMultiphaseDroplet
from .multiphase_droplet_top import InitialiseMultiphaseDropletTop
from .multiphase_droplet_variable_radius import InitialiseMultiphaseDropletVariableRadius
from .multiphase_lateral_bubble import InitialiseMultiphaseLateralBubble
from .wetting import InitialiseWetting
from .wetting_chemical_step import InitialiseWettingChemicalStep
from .init_from_file import InitialiseFromFile

# Backward-compatible alias — existing code imports ``Initialise``
Initialise = StandardInitialisation
