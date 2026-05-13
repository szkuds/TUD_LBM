"""Central configuration flags for TUD-LBM.

Edit these flags to control framework behaviour across the entire package.
All other config modules import from here.
"""

from pathlib import Path

# Configuration flags
ENABLE_X64 = True  # Enable 64-bit precision for JAX arrays
DISABLE_JIT = False  # Set to True for debugging (disables JIT compilation)
DEBUG_FLAG = False  # Set to True to enable debug output (jax.debug.print calls)

#: Default directory for storing simulation_type results
BASE_RESULTS_DIR: str = str(Path("~/TUD_LBM_data/results").expanduser())
