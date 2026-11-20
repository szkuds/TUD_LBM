r"""Well-balanced equilibrium that carries a pressure — pure function.

The equilibrium of Guo's well-balanced model (Phys. Fluids 33, 031709, 2021) in
its general form, with the pressure ``p`` in the second moment. For the moving
directions (``i > 0``):

.. math::

    f_i^{\\text{eq}} = w_i \\frac{p}{c_s^2} + w_i \\rho \\left[
        \\frac{\\mathbf{c}_i \\cdot \\mathbf{u}}{c_s^2}
        + \\frac{(\\mathbf{c}_i \\cdot \\mathbf{u})^2}{2 c_s^4}
        - \\frac{\\mathbf{u} \\cdot \\mathbf{u}}{2 c_s^2}
    \\right]

and the rest direction by mass conservation, ``feq_0 = rho - Σ_{i>0} feq_i``. Its
second moment is ``p I + rho u u``. ``"wb"`` is this equilibrium at ``p = 0``.

``p`` is not the fluid's pressure. It is an artificial one that the force gives
back: the multiphase macroscopic operator supplies ``p = rho_0 (mu - mu_b)`` and
reduces the interparticle force to ``-(rho - rho_0) grad(mu)``, so the momentum
equation is unchanged (Zhang et al., Int. J. Multiphase Flow 198, 105668, 2026,
Eq. 2 with ``rho_0`` the vapour density). The source term is the ``wb`` one, which
is that paper's Eq. 3. The third-order velocity term of its Eq. 2 is left out, as
in ``"wb"``: D2Q9 does not carry the moments it corrects.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.equilibrium._equilibrium_well_balanced import compute_equilibrium as compute_equilibrium_wb
from src.registry import equilibrium_operator

if TYPE_CHECKING:
    from src.lattice import Lattice

_CS2 = 1.0 / 3.0


@equilibrium_operator(name="wb_hydro_inc", source_term="wb", reference_density=True)
def compute_equilibrium(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    lattice: Lattice,
    pressure: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute the well-balanced equilibrium with *pressure* in its second moment.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, nz, 1, d)``.
        lattice: :class:`~setup.lattice.Lattice` with weights ``w``
            and velocity vectors ``c``.
        pressure: The pressure the second moment carries, shape
            ``(nx, ny, nz, 1, 1)``. ``None`` is no pressure: the ``"wb"`` equilibrium.

    Returns:
        Equilibrium populations ``feq``, shape ``(nx, ny, nz, q, 1)``.
    """
    feq = compute_equilibrium_wb(rho, u, lattice)
    if pressure is None:
        return feq
    moving = lattice.w[..., 1:, :] * pressure / _CS2  # (nx, ny, nz, q - 1, 1)
    rest = -jnp.sum(moving, axis=-2, keepdims=True)
    return feq + jnp.concatenate([rest, moving], axis=-2)
