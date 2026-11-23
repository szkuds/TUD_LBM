"""Wetting operators — the wall applicator, the contact-angle measurements and the hysteresis optimiser.

Nothing here is selected by configuration, so the package registers no kind and
has no factory. Its modules are imported directly by the wetting addon
(:mod:`src.addons._wetting`), which wires them into a run:

* ``build_wetting_applicator`` (_applicator.py) — the wall ghost-cell rewrite;
* ``compute_contact_angle`` (_contact_angle.py) and
  ``compute_contact_line_location`` (_contact_line.py) — the measurements;
* ``update_wetting_state`` / ``update_wetting_state_chemical_step``
  (_hysteresis.py) — the per-step optimiser;
* ``WettingParams`` / ``WettingState`` (_params.py) — the wall parameters and the
  addon's substate, ``State.ext["wetting"]``.

Example:
    from src.operators.wetting._contact_angle import compute_contact_angle

    ca_left, ca_right = compute_contact_angle(rho, rho_mean, edge=Edge.BOTTOM)
"""
