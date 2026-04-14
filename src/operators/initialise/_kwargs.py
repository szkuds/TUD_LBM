"""Private helpers for building initialiser keyword arguments."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.simulation_config import SimulationSetup


def _build_init_kwargs(
    setup: SimulationSetup,
    init_type: str,
    caller_kwargs: dict | None,
) -> dict:
    """Assemble keyword arguments for a population initialiser.

    Merges multiphase parameters from *setup*, applies caller overrides,
    and resolves the ``npz_path`` for ``init_from_file`` initialisers.

    Args:
        setup: The simulation setup containing multiphase parameters
            and config paths.
        init_type: The name of the initialisation scheme
            (e.g. ``"standard"``, ``"init_from_file"``).
        caller_kwargs: Optional caller-provided overrides that take
            precedence over setup-derived values.

    Returns:
        A dictionary of keyword arguments ready to pass to the
        initialiser function.
    """
    kw: dict = {}

    mp = setup.multiphase_params
    if mp is not None:
        kw.update(
            rho_l=mp.rho_l,
            rho_v=mp.rho_v,
            interface_width=mp.interface_width,
        )

    if caller_kwargs:
        kw.update(caller_kwargs)

    if init_type == "init_from_file" and "npz_path" not in kw:
        init_dir = setup.config.init_dir
        if init_dir is not None:
            kw["npz_path"] = init_dir

    return kw

