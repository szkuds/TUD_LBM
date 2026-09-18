Lattice and Array Conventions
=============================

The lattice fixes the discrete velocity set every operator works against,
and the array layout that flows through the whole pipeline.

The ``Lattice`` pytree
----------------------

:class:`~src.lattice.lattice.Lattice` is a :class:`typing.NamedTuple`, which
makes it a valid JAX pytree — it passes through ``jax.jit`` and
``jax.lax.scan`` without conversion.  All array fields are ``jax.Array``.

.. code-block:: python

    from src.lattice import build_lattice

    lattice = build_lattice("D2Q9")
    lattice.d              # 2
    lattice.q              # 9
    lattice.c.shape        # (1, 1, 1, 9, 2)
    lattice.w.shape        # (1, 1, 1, 9, 1)

===================  ========================  =====================================
Field                Shape                     Meaning
===================  ========================  =====================================
``name``             —                         ``"D2Q9"``, ``"D3Q19"``
``d``                —                         spatial dimensions
``q``                —                         discrete velocities
``c``                ``(1, 1, 1, q, d)``       velocity vectors
``w``                ``(1, 1, 1, q, 1)``       quadrature weights
``opp_indices``      ``(q,)``                  index of the opposite direction
``main_indices``     ``(m,)``                  cardinal (non-diagonal) directions
``right_indices``    ``(nr,)``                 directions with ``cx > 0``
``left_indices``     ``(nl,)``                 directions with ``cx < 0``
``top_indices``      ``(nt,)``                 directions with ``cy > 0``
``bottom_indices``   ``(nb,)``                 directions with ``cy < 0``
``front_indices``    ``(nf,)`` or ``None``     directions with ``cz > 0`` (3-D only)
``back_indices``     ``(nbk,)`` or ``None``    directions with ``cz < 0`` (3-D only)
===================  ========================  =====================================

The directional index sets are what boundary conditions and wetting operators
use to pick out the populations pointing into or out of a wall.  They are 1-D
and encode a direction → ``q``-index mapping only, with no spatial
assumptions baked in.

The 5-D array convention
------------------------

Every field array in the package has the shape
``(nx, ny, nz, q_or_1, d_or_1)``:

===========  ========================  ===============================
Array        Shape                     Notes
===========  ========================  ===============================
``f``        ``(nx, ny, nz, q, 1)``    population distributions
``rho``      ``(nx, ny, nz, 1, 1)``    density
``u``        ``(nx, ny, nz, 1, d)``    velocity
``force``    ``(nx, ny, nz, 1, d)``    force fields
``c``        ``(1, 1, 1, q, d)``       lattice velocities
``w``        ``(1, 1, 1, q, 1)``       lattice weights
===========  ========================  ===============================

The two trailing axes exist purely so that lattice arrays broadcast against
field arrays without reshaping at every call site — for example ``f * c``
contracts populations against velocities directly.

For 2-D simulations ``nz == 1``.  ``SimulationConfig`` normalises
``grid_shape`` to three entries in ``__post_init__``, so ``(64, 64)`` and
``(64, 64, 1)`` describe the same domain and downstream code never has to
branch on dimensionality.

Supported velocity models
-------------------------

D2Q9
~~~~

Nine velocities on a square lattice: the rest direction, four cardinal
neighbours with weight ``1/9``, and four diagonals with weight ``1/36``; the
rest weight is ``4/9``.  This is the default (``lattice_type = "D2Q9"``) and
the only lattice the MRT collision operator supports.

D3Q19
~~~~~

Nineteen velocities: rest (weight ``1/3``), six face neighbours (``1/18``),
and twelve edge neighbours (``1/36``).

Both builders compute ``opp_indices`` by searching for ``−cᵢ`` in the
velocity list rather than hard-coding a permutation, so the index tables
cannot drift out of step with the velocity ordering.

Building a lattice
------------------

.. code-block:: python

    from src.lattice import build_lattice

    lattice = build_lattice("d2q9")   # name is upper-cased before lookup

:func:`~src.lattice.lattice.build_lattice` resolves the name through the
registry's ``lattice`` kind and raises ``ValueError`` listing the supported
models on a miss.  ``SimulationConfig`` validates ``lattice_type`` against
the same registry in ``__post_init__``, so a typo is caught when the
configuration is loaded rather than mid-run — but that check is
case-sensitive.  ``lattice_type = "d2q9"`` is rejected by the configuration
even though ``build_lattice("d2q9")`` would succeed; use the registered
spelling in config files.

Adding a velocity model
-----------------------

Register a zero-argument builder that returns a fully populated
:class:`~src.lattice.lattice.Lattice`:

.. code-block:: python

    # src/lattice/lattice.py
    from src.registry import lattice_operator

    @lattice_operator(name="D3Q27", dim=3, q=27)
    def _build_d3q27() -> Lattice:
        c = jnp.array(...)[None, None, None, :, :]   # (1, 1, 1, 27, 3)
        w = jnp.array(...)[None, None, None, :, None] # (1, 1, 1, 27, 1)
        ...
        return Lattice(name="D3Q27", d=3, q=27, c=c, w=w, ...)

The registered ``name`` must be upper-case, since ``build_lattice`` upper-cases
its argument before the lookup.  The ``dim`` and ``q`` metadata are stored on
the registry entry and used by the CLI listings.
