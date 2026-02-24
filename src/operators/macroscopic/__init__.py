"""Macroscopic field operators for LBM simulations.

Computes macroscopic quantities (density, velocity) from the distribution
functions, with additional equations of state for multiphase simulations.

Classes:
    Macroscopic: Standard single-phase macroscopic field calculator.
    MacroscopicMultiphaseDW: Multiphase macroscopic fields (Dubble-well potential EOS).
    MacroscopicMultiphaseCS: Multiphase macroscopic fields (Carnahan-Starling EOS).
"""

from .macroscopic import Macroscopic
from .macroscopic_multiphase_dw import MacroscopicMultiphaseDW
from .macroscopic_multiphase_cs import MacroscopicMultiphaseCS
