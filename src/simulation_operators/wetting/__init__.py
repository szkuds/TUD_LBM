"""Wetting and contact angle models for LBM simulations.

Provides implementations for contact angle measurement, contact line tracking,
and wetting boundary conditions for multiphase simulations.

Classes:
    ContactAngle: Computes contact angles from the density field.
    ContactLineLocation: Computes contact line locations from the density field..
    WettingParameters: Configuration container for wetting properties.

Functions:
    determine_padding_modes: Determines padding modes for wetting boundaries.
    wetting_1d: Applies 1D wetting boundary condition.
    apply_wetting_to_all_edges: Applies wetting to all simulation_domain edges.
    has_wetting_bc: Checks if wetting boundary conditions are enabled.
"""

from .contact_angle import ContactAngle
from .contact_line_location import ContactLineLocation
from .wetting_util import determine_padding_modes, wetting_1d, apply_wetting_to_all_edges, has_wetting_bc, WettingParameters
