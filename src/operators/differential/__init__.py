"""Differential operators — gradient, Laplacian, and wetting-aware variants.

Public API
~~~~~~~~~~
Types:
    DifferentialConfig          — transient config for the factory
    DifferentialOperators       — result NamedTuple of pre-built callables

Functions:
    build_differential_fn(scheme)           — single operator lookup
    build_differential_operators(cfg)       — composite builder for simulation setup

Implementation modules (_gradient.py, _laplacian.py, _gradient_wetting.py,
_laplacian_wetting.py) are internal.  Auto-discovered by the loader; no
hardcoded import list.

Example:
    from operators.differential import build_differential_fn

    grad_fn = build_differential_fn("gradient")
    result  = grad_fn(grid, w, c, pad_mode)
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any
from typing import NamedTuple
import jax.numpy as jnp
from operators._loader import auto_load_operators
from operators.factory import build_operator
from operators.protocols import DifferentialOperator

# Auto-discover _gradient.py, _laplacian.py, _gradient_wetting.py,
# _laplacian_wetting.py to trigger their @register_operator decorators.
auto_load_operators("operators.differential")


# ══════════════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════════════


class DifferentialConfig(NamedTuple):
    """Transient configuration consumed by :func:`build_differential_operators`.

    Created inside :func:`~setup.simulation_setup.build_setup` and
    discarded after the factory call — it is **never** stored on
    :class:`~setup.simulation_setup.SimulationSetup`.

    Attributes:
        w:              Lattice weights, shape ``(q,)``.
        c:              Lattice velocity vectors, shape ``(2, q)``.
        pad_modes:      Four padding-mode strings
                        ``[right_y, left_y, bottom_x, top_x]``.
        wetting_params: ``None`` when wetting is disabled; otherwise the
                        dict accepted by :func:`make_wetting_gradient`.
        chemical_step:  Optional step index for chemical-step wetting
                        geometries.
        bc_config:      Boundary-condition config dict, e.g.
                        ``{"bottom": "wetting", "top": "bounce-back"}``.
                        Passed through to :func:`apply_wetting_to_all_edges`.
    """

    w: jnp.ndarray
    c: jnp.ndarray
    pad_modes: list[str]
    wetting_params: dict[str, Any] | None = None
    chemical_step: int | None = None
    bc_config: dict[str, Any] | None = None

    @property
    def wetting_enabled(self) -> bool:
        """Return ``True`` when wetting parameters are present."""
        return self.wetting_params is not None

    @property
    def chemical_step_enabled(self) -> bool:
        """Return ``True`` when chemical step parameters are present."""
        return self.chemical_step is not None


class DifferentialOperators(NamedTuple):
    """Pre-built, jitted differential operators ready for the simulation loop.

    All three callables have signature
    ``f(grid: jnp.ndarray) -> jnp.ndarray`` so the loop never needs to
    know about pad_modes, wetting, or BCs.

    .. note::

       This NamedTuple contains Python *callables*, not JAX array leaves.
       Because :class:`SimulationSetup` is closed over (not passed through
       ``lax.scan``), this is correct and intentional.

    Attributes:
        grad_standard: Standard LBM-stencil gradient.  **Always** built
            from pad_modes only, independent of wetting.
            Use this for chemical_potential.
        grad_field: Gradient for density / order-parameter fields.
            Identical to *grad_standard* when wetting is off; includes
            ghost-cell correction when wetting is on.
        laplacian: LBM-stencil Laplacian.
            Wetting-aware when ``cfg.wetting_enabled`` is ``True``;
            standard otherwise.
    """

    grad_standard: Callable[[jnp.ndarray], jnp.ndarray]
    grad_field: Callable[[jnp.ndarray], jnp.ndarray]
    laplacian: Callable[[jnp.ndarray], jnp.ndarray]


# ══════════════════════════════════════════════════════════════════════════════
# Factory functions
# ══════════════════════════════════════════════════════════════════════════════


def build_differential_fn(scheme: str) -> DifferentialOperator:
    """Return a differential operator by scheme name.

    Delegates to the central ``build_operator()`` factory.

    Args:
        scheme: ``"gradient"``, ``"laplacian"``, ``"gradient_wetting"``,
                or ``"laplacian_wetting"``.

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

    pad = tuple(cfg.pad_modes)

    # ── Base operators (always resolved from registry) ───────────
    _gradient = build_differential_fn("gradient")
    _laplacian = build_differential_fn("laplacian")

    @jax.jit
    def grad_standard(grid: jnp.ndarray) -> jnp.ndarray:
        return _gradient(grid, cfg.w, cfg.c, pad)

    @jax.jit
    def laplacian_standard(grid: jnp.ndarray) -> jnp.ndarray:
        return _laplacian(grid, cfg.w, pad)

    # ── Wetting branch ───────────────────────────────────────────
    if cfg.wetting_enabled:
        _wetting_grad_builder = build_differential_fn("gradient_wetting")
        _wetting_grad = _wetting_grad_builder(cfg.w, cfg.c, pad, cfg.bc_config)

        _wetting_lap_builder = build_differential_fn("laplacian_wetting")
        _wetting_lap = _wetting_lap_builder(cfg.w, pad, cfg.bc_config)

        ws = cfg.wetting_params

        @jax.jit
        def grad_field(grid: jnp.ndarray) -> jnp.ndarray:
            return _wetting_grad

        @jax.jit
        def laplacian(grid: jnp.ndarray) -> jnp.ndarray:
            return _wetting_lap
    else:
        grad_field = grad_standard  # same object, zero overhead
        laplacian = laplacian_standard

    return DifferentialOperators(
        grad_standard=grad_standard,
        grad_field=grad_field,
        laplacian=laplacian,
    )


__all__ = [
    "DifferentialConfig",
    "DifferentialOperators",
    "build_differential_fn",
    "build_differential_operators",
]
