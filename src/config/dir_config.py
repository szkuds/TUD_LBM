"""Directory configuration constants for TUD-LBM simulations.

This module contains all directory-related configuration constants.
"""

from pathlib import Path

# =============================================================================
# Directory Constants
# =============================================================================

#: Default directory for storing simulation_type results
BASE_RESULTS_DIR: str = str(Path("~/TUD_LBM_data/results").expanduser())
