"""Phase-masked gravity force module.

Provides a constant body force restricted to the **dispersed** phase — the
liquid for a droplet run, the vapour for a bubble run. Restricting it matters
in a periodic domain: a force spread over the continuous phase accelerates the
whole ambient body instead of driving the inclusion against the wall.

Which phase is dispersed is a property of the *initial* topology, so it is
resolved once at build time (see :func:`_resolve_dispersed`) rather than
detected per step. Step-time detection — as
:func:`~src.operators.wetting._interface_crossings.detect_bubble` does for the
hysteresis knobs — reads a single wall row, and collapses to "droplet" the
moment the dispersed phase leaves that wall. For a wetting knob that is a
recoverable, local mistake; for a body force it would silently switch the
momentum injection to the entire continuous phase.

Usage::

    # Via registry (preferred)
    from operators.force import build_force_fn

    module = build_force_fn("gravity_masked_force")
    pre = module.build({"force_g": 0.001}, (64, 64), config=config, lattice=lattice)
    force = module.compute(state, pre)

    # Direct (internal / testing)
    from operators.force._gravity_masked import GravityForceModule

    pre = GravityForceModule.build({"force_g": 0.001}, (64, 64), config=config, lattice=lattice)
    force = GravityForceModule.compute(state, pre)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple
import jax.numpy as jnp
from src.operators.force._gravity import _build_gravity_template
from src.registry import force_model

if TYPE_CHECKING:
    from src.pipeline.state import State

#: Accepted values of the ``dispersed`` key, matching the vocabulary of
#: ``[initialisation]`` and the ``multiphase_bubbles`` initialiser.
_DISPERSED_PHASES = ("liquid", "vapour")

#: Phase driven when nothing in the config declares a topology. Liquid keeps the
#: single-phase and droplet behaviour this module had before bubbles existed.
_FALLBACK_DISPERSED = "liquid"

#: Topology implied by initialisers that fix it themselves, used when
#: ``[initialisation]`` carries no ``dispersed`` key. ``multiphase_bubbles``
#: takes the key but defaults to vapour, and ``multiphase_bubble_top`` always
#: places a vapour inclusion — see :mod:`src.operators.initialise`.
_INIT_TYPE_DISPERSED = {
    "multiphase_bubbles": "vapour",
    "multiphase_bubble_top": "vapour",
}


def _resolve_dispersed(params: dict, config: object | None) -> str:
    """Resolve which phase the masked body force acts on.

    Precedence, first match wins:

    1. ``dispersed`` in the ``[gravity_masked_force]`` section — the explicit
       override, for runs whose initialiser does not declare a topology
       (``init_from_file``, for instance).
    2. ``dispersed`` in ``[initialisation]`` — what the initialiser was told.
    3. The topology fixed by ``init_type`` (:data:`_INIT_TYPE_DISPERSED`).
    4. :data:`_FALLBACK_DISPERSED`.

    Args:
        params: Config dict from the ``[gravity_masked_force]`` TOML section.
        config: The :class:`~src.config.SimulationConfig`, or None.

    Returns:
        Either ``"liquid"`` or ``"vapour"``.

    Raises:
        ValueError: If a declared value is neither ``"liquid"`` nor ``"vapour"``.
    """
    declared = params.get("dispersed")
    if declared is None and config is not None:
        # getattr is intentional here: config is typed as object from **kwargs.
        init = getattr(config, "initialisation", None)
        if isinstance(init, dict):
            declared = init.get("dispersed")
        init_type = getattr(config, "init_type", None)
        if declared is None and isinstance(init_type, str):
            declared = _INIT_TYPE_DISPERSED.get(init_type)
    if declared is None:
        return _FALLBACK_DISPERSED

    dispersed = str(declared).strip().lower()
    if dispersed not in _DISPERSED_PHASES:
        msg = f"'dispersed' must be one of {_DISPERSED_PHASES}, got {declared!r}."
        raise ValueError(msg)
    return dispersed


class GravityPrecomputed(NamedTuple):
    """Container for gravity precomputed data.

    Attributes:
        template: constant force field, shape (nx, ny, nz, 1, d)
        rho_l: liquid-phase reference density or None
        rho_v: vapor-phase reference density or None
        dispersed: phase the force acts on — ``"liquid"`` or ``"vapour"``
    """

    template: jnp.ndarray
    rho_l: float | None
    rho_v: float | None
    dispersed: str = _FALLBACK_DISPERSED


# ══════════════════════════════════════════════════════════════════════
# ForceOperator protocol — registry-backed module
# ══════════════════════════════════════════════════════════════════════


@force_model(name="gravity_masked_force")
class GravityForceModule:
    """Gravity force conforming to :class:`ForceOperator` protocol.

    Stateless — it uses the default no force state hooks.
    """

    @staticmethod
    def build(
        params: dict,
        grid_shape: tuple[int, ...],
        **kwargs: object,
    ) -> GravityPrecomputed:
        """Build a constant gravity-force template.

        Args:
            params: Config dict from ``[gravity_masked_force]`` TOML section.
                Required key: ``force_g``.
                Optional keys: ``inclination_angle_deg`` (default 0) and
                ``dispersed`` (``"liquid"`` / ``"vapour"``, overriding the
                topology inferred from ``[initialisation]``).
            grid_shape: Spatial dimensions ``(nx, ny, nz, ...)``.
            **kwargs: Additional arguments including ``lattice`` (for dimension info).

        Returns:
            Gravity template array, shape ``(nx, ny, nz, 1, d)``.
        """
        template = _build_gravity_template(params, grid_shape, **kwargs)

        # Extract optional reference densities from config (if provided).
        # getattr is intentional here: config is typed as object from **kwargs.
        config = kwargs.get("config")
        _rho_l_raw = getattr(config, "rho_l", None) if config is not None else None
        _rho_v_raw = getattr(config, "rho_v", None) if config is not None else None
        rho_l = float(_rho_l_raw) if _rho_l_raw is not None else None
        rho_v = float(_rho_v_raw) if _rho_v_raw is not None else None

        return GravityPrecomputed(
            template=template,
            rho_l=rho_l,
            rho_v=rho_v,
            dispersed=_resolve_dispersed(params, config),
        )

    @staticmethod
    def compute(
        state: State,
        precomputed: GravityPrecomputed,
        **_kwargs: object,
    ) -> jnp.ndarray:
        """Compute gravity force (step-time, jittable).

        Args:
            state: Current simulation :class:`State`. Only ``state.f``
                is used (to compute density).
            precomputed: Gravity template from :meth:`build`.
            **kwargs: Additional arguments (ignored).

        Returns:
            Gravity force field, shape ``(nx, ny, nz, 1, d)``, non-zero only
            over the dispersed phase.
        """
        rho = jnp.sum(state.f, axis=-2, keepdims=True)

        # If both liquid and vapor reference densities are available, compute
        # a smooth mask between phases and apply it to the gravity force.
        if precomputed.rho_l is not None and precomputed.rho_v is not None:
            # 1 in the liquid, 0 in the vapour, linear across the interface.
            liquid = jnp.clip(
                (rho - precomputed.rho_v) / (precomputed.rho_l - precomputed.rho_v),
                0.0,
                1.0,
            )
            # The force drives the dispersed phase, so a bubble run — vapour
            # dispersed in liquid — takes the complementary mask. ``dispersed``
            # is a build-time string, so the branch resolves at trace time and
            # only one mask reaches the graph.
            mask = liquid if precomputed.dispersed == "liquid" else 1.0 - liquid
            return -precomputed.template * rho * mask

        # Fallback: single-phase (no mask)
        return -precomputed.template * rho
