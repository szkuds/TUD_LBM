"""Buoyancy-referenced gravity force module.

Provides a body force weighted by the density *excess over the vapour*,
``(rho - rho_v) * g``, rather than by the local ``rho``. Subtracting the
vapour reference is what makes it usable in a periodic or wall-bounded
domain: the ambient vapour of a droplet run is left force-free, so the whole
ambient body is not accelerated along with the inclusion.

Written this way there is no topology branch. A liquid droplet in vapour is
driven at ``drho * g`` and its ambient at zero; a vapour bubble in liquid is
itself force-free and the *liquid* around it carries ``drho * g``, so the
bubble rises on the resulting pressure gradient with no sign flag anywhere in
the config. Both are the same formula.

Why the force must sit on the dense phase, not the inclusion
------------------------------------------------------------
The predecessor of this module injected the *net* buoyancy ``+-drho * g``
masked to the dispersed phase, arguing that a force-free continuous phase
builds no hydrostatic gradient and therefore nothing pushes back. That is
self-consistent for a droplet and self-defeating for a bubble: with nothing
pushing back, the light phase has to carry the entire momentum injection in
its own inertia. A body force enters as ``du = F / rho``
(:mod:`src.operators.macroscopic._multiphase`), so masking ``drho * g`` onto
the vapour accelerates it at ``drho / rho_v`` times ``g`` — a factor of 800
for ``rho_l = 12.18``, ``rho_v = 0.015``, against ``drho / rho_l ~ 1`` for the
droplet the formulation was designed around.

Nor does it reach a steady state. Balancing that body force hydrostatically
needs a pressure difference ``drho * g * 2R`` across the inclusion, which for
the parameters above is 82x the vapour's *absolute* pressure. No equilibrium
exists, so the gas evacuates; and because the injected force did not scale
with ``rho``, thinning the gas raised ``F / rho`` further. Measured: velocity
0.045 -> 0.073 -> 0.11 -> 0.36 over 300 steps while the minimum density fell
by a decade.

The dense phase has no such limit — it sustains a pressure gradient, in
tension if need be — which is why the weight references ``rho_v`` and not
``rho_l``. Buoyancy on the inclusion is then not injected at all: it emerges
as the reaction to the surrounding liquid's hydrostatic gradient, which is
also what supplies the added mass that bounds the rise speed.

The weight is continuous through the interface, so unlike a hard threshold at
``rho_mean`` it has no tie-breaking rule and no discontinuity. The net force
over a droplet is unchanged, ``drho * g * A``, because a tanh interface is
symmetric about ``rho_mean`` and its two halves cancel.

Boundary conditions are load-bearing
------------------------------------
The liquid can only stay at rest if it can hold a hydrostatic ramp along every
direction that has a gravity component, and a periodic axis cannot support a
linear pressure ramp. Under an inclined gravity the tangential component is
then balanced by wall shear instead, and the domain fills with Poiseuille flow
— for a 201x101 channel at 50 degrees, ``u_max = 3e-3``, some 230x the
bubble's own buoyant slide speed. Close the tangential axis (``left`` and
``right`` set to ``bounce-back``) and the same component is balanced by a
density ramp of about 1% of ``rho_l`` instead, with the liquid at rest. Do not
use ``symmetry`` for those walls: its correction assumes a periodic x.

Switching gravity on abruptly leaves the fluid one acoustic crossing behind
the force, which launches a standing sound wave that rings for tens of
thousands of steps at an amplitude far above the motion being measured. The
optional ``ramp_steps`` / ``ramp_start_t`` pair brings the force up slowly
enough that the fluid stays quasi-statically balanced and no wave is excited.
``ramp_start_t`` is an *absolute* timestep because ``state.t`` survives
restarts (:func:`src.pipeline.runner._t_from_snapshot` reads it back from the
snapshot filename), so a resumed run must name the step at which gravity was
first applied for every chunk to agree on the same ramp.

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


class GravityPrecomputed(NamedTuple):
    """Container for gravity precomputed data.

    Attributes:
        template: constant force field, shape (nx, ny, nz, 1, d)
        rho_v: vapour-phase reference density, or None for a single-phase run
        ramp_steps: length of the linear switch-on ramp, or None for no ramp
        ramp_start_t: absolute timestep at which the ramp begins
    """

    template: jnp.ndarray
    rho_v: float | None
    ramp_steps: float | None = None
    ramp_start_t: float = 0.0


def _ramp_fraction(t: jnp.ndarray, precomputed: GravityPrecomputed) -> jnp.ndarray | float:
    """Fraction of the full force in effect at timestep ``t``.

    ``ramp_steps`` is a build-time value, so an unramped run resolves to the
    Python float ``1.0`` at trace time and puts nothing in the graph.
    """
    if precomputed.ramp_steps is None:
        return 1.0
    return jnp.clip((t - precomputed.ramp_start_t) / precomputed.ramp_steps, 0.0, 1.0)


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
                Optional keys: ``inclination_angle_deg`` (default 0),
                ``ramp_steps`` (default None, meaning the force is applied at
                full strength immediately) and ``ramp_start_t`` (default 0).
            grid_shape: Spatial dimensions ``(nx, ny, nz, ...)``.
            **kwargs: Additional arguments including ``lattice`` (for dimension info).

        Returns:
            A :class:`GravityPrecomputed` holding the constant template, the
            vapour reference density (``None`` when the config carries none)
            and the resolved switch-on ramp.

        Raises:
            ValueError: If ``ramp_steps`` is set to a non-positive value.
        """
        template = _build_gravity_template(params, grid_shape, **kwargs)

        # Extract the optional vapour reference from config (if provided).
        # getattr is intentional here: config is typed as object from **kwargs.
        config = kwargs.get("config")
        _rho_v_raw = getattr(config, "rho_v", None) if config is not None else None

        ramp_steps = params.get("ramp_steps")
        if ramp_steps is not None and float(ramp_steps) <= 0.0:
            msg = f"'ramp_steps' must be positive, got {ramp_steps!r}. Omit the key for no ramp."
            raise ValueError(msg)

        return GravityPrecomputed(
            template=template,
            rho_v=float(_rho_v_raw) if _rho_v_raw is not None else None,
            ramp_steps=float(ramp_steps) if ramp_steps is not None else None,
            ramp_start_t=float(params.get("ramp_start_t", 0.0)),
        )

    @staticmethod
    def compute(
        state: State,
        precomputed: GravityPrecomputed,
        **_kwargs: object,
    ) -> jnp.ndarray:
        """Compute the buoyancy-referenced body force (step-time, jittable).

        Args:
            state: Current simulation :class:`State`. Uses ``state.f`` (to
                compute density) and ``state.t`` (for the switch-on ramp).
            precomputed: Gravity template from :meth:`build`.
            **_kwargs: Additional arguments (ignored).

        Returns:
            Body force field, shape ``(nx, ny, nz, 1, d)``, weighted by the
            density excess over the vapour: ``drho * g`` in the liquid,
            vanishing in the vapour.
        """
        rho = jnp.sum(state.f, axis=-2, keepdims=True)

        # Single-phase runs carry no reference density to subtract, leaving the
        # plain local weight rho*g.
        weight = rho if precomputed.rho_v is None else rho - precomputed.rho_v

        return -precomputed.template * _ramp_fraction(state.t, precomputed) * weight
