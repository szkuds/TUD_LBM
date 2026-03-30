"""Differential operators — gradient, Laplacian, and wetting-aware variants.

Public API:
    build_differential_fn(scheme)           — single operator lookup
    build_differential_operators(cfg)       — composite builder for simulation setup

Implementation modules (_gradient.py, _laplacian.py, _gradient_wetting.py)
are internal.  Auto-discovered by the loader; no hardcoded import list.

Example:
    from operators.differential import build_differential_fn

    grad_fn = build_differential_fn("gradient")
    result  = grad_fn(grid, w, c, pad_mode)
"""

from __future__ import annotations

from operators.protocols import DifferentialOperator
from operators.factory import build_operator
from operators._loader import auto_load_operators
from operators.differential.config import DifferentialConfig
from operators.differential.operators import DifferentialOperators

# Auto-discover _gradient.py, _laplacian.py, _gradient_wetting.py
# to trigger their @register_operator decorators.
auto_load_operators("operators.differential")


def build_differential_fn(scheme: str) -> DifferentialOperator:
    """Return a differential operator by scheme name.

    Delegates to the central ``build_operator()`` factory.

    Args:
        scheme: ``"gradient"``, ``"laplacian"``, or ``"gradient_wetting"``.

    Returns:
        The raw operator function / closure builder from the registry.

    Raises:
        ValueError: If scheme is not registered.

    Example:
        >>> grad = build_differential_fn("gradient")
        >>> result = grad(grid, w, c, pad_mode)
    """
    return build_operator("differential", scheme)


def build_differential_operators(cfg: DifferentialConfig) -> DifferentialOperators:
    """Build all pre-compiled differential operators from *cfg*.

    Called **once** at setup time.  Resolves base operators from the
    registry via ``build_differential_fn``, binds config into closures so
    every callable in the returned NamedTuple has signature
    ``(grid) → array``, matching the ``DifferentialOperator`` protocol.

    ``grad_standard`` is always the pad-modes-only gradient, independent
    of wetting.  ``grad_field`` is wetting-corrected when
    ``cfg.wetting_enabled`` is ``True``, otherwise aliased to
    ``grad_standard``.

    Args:
        cfg: A :class:`DifferentialConfig` with lattice weights,
             velocities, pad modes, and optional wetting parameters.

    Returns:
        :class:`DifferentialOperators` NamedTuple of
        ``Callable[[jnp.ndarray], jnp.ndarray]``.
    """
    import jax
    import jax.numpy as jnp

    pad = tuple(cfg.pad_modes)

    # ── Base operators (always resolved from registry) ───────────
    _gradient = build_differential_fn("gradient")
    _laplacian = build_differential_fn("laplacian")

    @jax.jit
    def grad_standard(grid: jnp.ndarray) -> jnp.ndarray:
        return _gradient(grid, cfg.w, cfg.c, pad)

    @jax.jit
    def laplacian(grid: jnp.ndarray) -> jnp.ndarray:
        return _laplacian(grid, cfg.w, pad)

    # ── Wetting branch ───────────────────────────────────────────
    if cfg.wetting_enabled:
        _wetting_builder = build_differential_fn("gradient_wetting")
        _wetting_grad = _wetting_builder(cfg.w, cfg.c, pad, cfg.bc_config)

        # Adapter: read dynamic wetting state from the mutable
        # wetting_params dict so grad_field keeps the 1-arg contract.
        ws = cfg.wetting_params

        @jax.jit
        def grad_field(grid: jnp.ndarray) -> jnp.ndarray:
            return _wetting_grad(
                grid,
                ws["phi_l"], ws["phi_r"],
                ws["d_rho_l"], ws["d_rho_r"],
                ws["rho_l"], ws["rho_v"],
                ws["width"],
            )
    else:
        grad_field = grad_standard  # same object, zero overhead

    return DifferentialOperators(
        grad_standard=grad_standard,
        grad_field=grad_field,
        laplacian=laplacian,
    )


__all__ = ["build_differential_fn", "build_differential_operators"]
