"""Central configuration flags for TUD-LBM.

Edit these flags to control framework behaviour across the entire package.
All other config modules import from here.
"""

import os
from pathlib import Path

# Configuration flags
ENABLE_X64: bool = True  # Enable 64-bit precision for JAX arrays
DISABLE_JIT: bool = False  # Set to True for debugging (disables JIT compilation)
DEBUG_FLAG_WETTING: bool = False  # Set to True to enable debug output (jax.debug.print calls)
DEBUG_FLAG_STABILITY: bool = False  # Set via --debug-stability; enables stability diagnostics

# Wetting-debug tunables (only read when DEBUG_FLAG_WETTING is True)
DEBUG_WETTING_INTERVAL: int = 50  # Timesteps between logged wetting rows; --debug-wetting-interval

# Stability-diagnostics tunables (only read when DEBUG_FLAG_STABILITY is True)
STABILITY_VAPOR_FRACTION: float = 0.2  # wake mask: rho < rho_v + frac * (rho_l - rho_v)
STABILITY_GRAD_RHO_FRACTION: float = 0.05  # exclude cells with |grad rho| > frac * (rho_l - rho_v)

#: Default directory for storing simulation_type results. Reads
#: TUD_LBM_DATA_DIR so DelftBlue jobs (whose ~ is quota-limited home, not
#: scratch) default to scratch instead — see scripts/setup_on_delftblue.sh
#: and scripts/db_job_template.sh.in, which export it.
BASE_RESULTS_DIR: str = str(Path(os.environ.get("TUD_LBM_DATA_DIR", "~/TUD_LBM_data")).expanduser())
