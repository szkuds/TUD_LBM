"""Macroscopic operators — implementations of MacroscopicOperator protocol.

Public API: build_macroscopic_fn(), MultiphaseParams, build_multiphase_params

Implementation modules (_single_phase.py, _multiphase.py) are internal; use the factory to access.

Example:
    from operators.macroscopic import build_macroscopic_fn, MultiphaseParams, build_multiphase_params

    macro = build_macroscopic_fn("standard")
    rho, u = macro(f, lattice)

    mp = build_multiphase_params(config)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast
from tud_lbm.operators._loader import auto_load_operators
from tud_lbm.operators.factory import build_operator
from tud_lbm.operators.macroscopic import eos as _eos  # noqa: F401

if TYPE_CHECKING:
    from tud_lbm.config import SimulationConfig
    from tud_lbm.operators.protocols import MacroscopicOperator


class MultiphaseParams(NamedTuple):
    """Equation-of-state and surface-tension parameters.

    All fields are Python scalars (compile-time constants).
    """

    eos: str
    kappa: float
    rho_l: float
    rho_v: float
    interface_width: int
    g: float | None = None
    a_eos: float | None = None
    b_eos: float | None = None
    r_eos: float | None = None
    t_eos: float | None = None


# Auto-discover and import private operator modules for registry registration
auto_load_operators("tud_lbm.operators.macroscopic")

# Import EOS subpackage to trigger EOS operator registration.


def build_macroscopic_fn(scheme: str = "standard") -> MacroscopicOperator:
    """Return a macroscopic operator satisfying MacroscopicOperator protocol.

    Args:
        scheme: Macroscopic model name ("standard" or others).
                Defaults to "standard" (single-phase density and velocity).

    Returns:
        A callable satisfying the MacroscopicOperator protocol.
        Can be called as: operator(f, lattice, force=None) → (rho, u)

        Type-checkers see this as a MacroscopicOperator, so:
            op: MacroscopicOperator = build_macroscopic_fn("standard")

        Type-checkers will verify any use of op matches the protocol.

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from operators.macroscopic import build_macroscopic_fn
        >>> macroscopic = build_macroscopic_fn("standard")
        >>> rho, u = macroscopic(f, lattice)
    """
    return cast("MacroscopicOperator", build_operator("macroscopic", scheme))  # type: ignore[type-var]


def build_multiphase_params(config: SimulationConfig) -> MultiphaseParams:
    """Construct :class:`MultiphaseParams` from a configuration object.

    Args:
        config: An object with multiphase attributes (``eos``, ``kappa``,
            ``rho_l``, ``rho_v``, ``interface_width``, …).

    Returns:
        A :class:`MultiphaseParams` NamedTuple.

    Raises:
        ValueError: If required multiphase fields are missing.
    """
    for name in ("eos", "kappa", "rho_l", "rho_v", "interface_width"):
        if getattr(config, name, None) is None:
            msg = f"'{name}' is required for multiphase simulations"
            raise ValueError(msg)

    return MultiphaseParams(
        eos=config.eos,
        kappa=config.kappa,
        rho_l=config.rho_l,
        rho_v=config.rho_v,
        interface_width=config.interface_width,
        g=getattr(config, "g", None),
        a_eos=getattr(config, "a_eos", None),
        b_eos=getattr(config, "b_eos", None),
        r_eos=getattr(config, "r_eos", None),
        t_eos=getattr(config, "t_eos", None),
    )


__all__ = [
    "MultiphaseParams",
    "build_macroscopic_fn",  # ← Primary API (use this!)
    "build_multiphase_params",
]
