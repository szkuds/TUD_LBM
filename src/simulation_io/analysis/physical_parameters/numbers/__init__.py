"""One module per dimensionless number, self-registered under kind ``dimensionless``.

Adding a number is adding a file here. Auto-discovery imports every ``_*.py``,
the decorator registers it, and it then appears as a ``regime-map`` axis, a
``compare --label-param`` legend term and a row in ``physical_parameters.txt``
without any list being edited.

Each module owns its formula *and* the presentation metadata for it; nothing
downstream carries a per-number branch.
"""

from src.operators._loader import auto_load_operators

auto_load_operators("src.simulation_io.analysis.physical_parameters.numbers")
