"""JAX configuration settings for TUD-LBM.

This module provides centralized JAX configuration options.
Import and call `configure_jax()` at the start of your script to apply settings.
"""

# Set environment variable to prevent JAX from pre-allocating all GPU memory.
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import jax

# Configuration flags
ENABLE_X64 = True  # Enable 64-bit precision for JAX arrays
DISABLE_JIT = False  # Set to True for debugging (disables JIT compilation)


def configure_jax(
    enable_x64: bool | None = None,
    disable_jit: bool | None = None,
) -> None:
    """Configure JAX settings.

    Call this function at the start of your script to apply JAX configuration.
    If parameters are None, the module-level defaults are used.

    Args:
        enable_x64: Enable 64-bit precision. Defaults to module-level ENABLE_X64.
        disable_jit: Disable JIT compilation for debugging. Defaults to module-level DISABLE_JIT.
    """
    x64 = enable_x64 if enable_x64 is not None else ENABLE_X64
    no_jit = disable_jit if disable_jit is not None else DISABLE_JIT

    jax.config.update("jax_enable_x64", x64)
    jax.config.update("jax_disable_jit", no_jit)
