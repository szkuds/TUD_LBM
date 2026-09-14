"""Human-readable physical parameter overview for a simulation run.

Dimensionless numbers are registry-driven: each one is a module under
``numbers/`` registered with kind ``dimensionless``, and importing this package
is what registers them. Consumers ask for them by key -- see
:func:`dimensionless_keys` and :func:`dimensionless_label`.
"""

from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs
from src.simulation_io.analysis.physical_parameters.length_scale_figure import write_length_scale_figure
from src.simulation_io.analysis.physical_parameters.physical_parameters import BondNumbers
from src.simulation_io.analysis.physical_parameters.physical_parameters import DimensionlessNumbers
from src.simulation_io.analysis.physical_parameters.physical_parameters import build_overview
from src.simulation_io.analysis.physical_parameters.physical_parameters import compute_bond_numbers
from src.simulation_io.analysis.physical_parameters.physical_parameters import compute_dimensionless_numbers
from src.simulation_io.analysis.physical_parameters.physical_parameters import dimensionless_keys
from src.simulation_io.analysis.physical_parameters.physical_parameters import dimensionless_label
from src.simulation_io.analysis.physical_parameters.physical_parameters import inclusion_mask_from_rho
from src.simulation_io.analysis.physical_parameters.physical_parameters import measure_init_phase_densities
from src.simulation_io.analysis.physical_parameters.physical_parameters import resolve_dimensionless_inputs
from src.simulation_io.analysis.physical_parameters.physical_parameters import write_physical_parameters

__all__ = [
    "BondNumbers",
    "DimensionlessInputs",
    "DimensionlessNumbers",
    "build_overview",
    "compute_bond_numbers",
    "compute_dimensionless_numbers",
    "dimensionless_keys",
    "dimensionless_label",
    "inclusion_mask_from_rho",
    "measure_init_phase_densities",
    "resolve_dimensionless_inputs",
    "write_length_scale_figure",
    "write_physical_parameters",
]
