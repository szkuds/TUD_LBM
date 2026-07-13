"""Central configuration flags for TUD-LBM.

Edit these flags to control framework behaviour across the entire package.
All other config modules import from here.
"""

from pathlib import Path

# Configuration flags
ENABLE_X64: bool = True  # Enable 64-bit precision for JAX arrays
DISABLE_JIT: bool = False  # Set to True for debugging (disables JIT compilation)
DEBUG_FLAG_WETTING: bool = False  # Set to True to enable debug output (jax.debug.print calls)
DEBUG_FLAG_STABILITY: bool = False  # Set via --debug-stability; enables stability diagnostics

# Stability-diagnostics tunables (only read when DEBUG_FLAG_STABILITY is True)
STABILITY_VAPOR_FRACTION: float = 0.2  # wake mask: rho < rho_v + frac * (rho_l - rho_v)
STABILITY_GRAD_RHO_FRACTION: float = 0.05  # exclude cells with |grad rho| > frac * (rho_l - rho_v)

#: Default directory for storing simulation_type results
BASE_RESULTS_DIR: str = str(Path("~/TUD_LBM_data/results").expanduser())
