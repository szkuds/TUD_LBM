"""Legend labels for cross-run comparison plots, built from dimensionless numbers.

A comparison figure overlays N runs, and a legend entry naming the run directory
says nothing about the physics separating the curves. These helpers label each
run with the dimensionless numbers already derived for its
``physical_parameters.txt`` -- and, by default, only with the ones that actually
*differ* across the set being compared, since a quantity every run shares is
noise in a legend.

Two collapses keep the automatic label short. Only one member of the Bond family
appears: ``Bo_parallel`` when the runs are inclined, plain ``Bo`` when they are
not. Only ``Re`` appears of the buoyancy pair, because ``Re = sqrt(Ar)`` carries
the same information. Both ``Bo_perp`` and ``Ar`` remain reachable by naming them
explicitly (``tud-lbm compare --label-param``), which also bypasses the
differing-only test.
"""

from __future__ import annotations
import math
from collections import Counter
from typing import TYPE_CHECKING
from src.simulation_io.plotting.figure_config import LABEL_AR
from src.simulation_io.plotting.figure_config import LABEL_BO
from src.simulation_io.plotting.figure_config import LABEL_BO_PAR
from src.simulation_io.plotting.figure_config import LABEL_BO_PERP
from src.simulation_io.plotting.figure_config import LABEL_OH
from src.simulation_io.plotting.figure_config import LABEL_RE

if TYPE_CHECKING:
    from collections.abc import Sequence
    from src.simulation_io.analysis.physical_parameters import DimensionlessNumbers

#: Key selecting the run's own name (``simulation_name`` or its directory)
#: instead of a number, so today's labelling stays available as an override.
NAME_KEY = "name"

#: Fractional spread a quantity must exceed before it counts as a sweep axis.
#:
#: Not a float-comparison epsilon. Bo/Oh/Re are built from a length and a density
#: contrast *measured off each run's own init snapshot*, so two runs that differ
#: only in something else (a contact-angle window, say) still land a few tenths of
#: a percent apart. Exact inequality read that noise as a real difference and put
#: an identical prefix on every legend entry. A quantity has to move by more than
#: this to be what distinguishes the runs; below it, the labels fall back to the
#: run names, and ``--label-param`` still forces the quantity in.
_SIGNIFICANT_SPREAD = 0.02


def _math_body(label: str) -> str:
    """The inner math of a ``$...$`` label from ``figure_config``.

    Terms are composed into ``$<body> = <value>$`` segments, so the spelling of
    each quantity stays owned by ``figure_config`` rather than duplicated here.
    """
    return label.strip("$")


#: Every quantity that may appear in a label, mapped to its math-text body.
#: Keys are attribute names on :class:`DimensionlessNumbers`.
_QUANTITY_LABELS: dict[str, str] = {
    "bo": _math_body(LABEL_BO),
    "bo_parallel": _math_body(LABEL_BO_PAR),
    "bo_perp": _math_body(LABEL_BO_PERP),
    "oh": _math_body(LABEL_OH),
    "ar": _math_body(LABEL_AR),
    "re": _math_body(LABEL_RE),
}

#: Valid ``--label-param`` values, in the order the CLI lists them.
LABEL_PARAM_CHOICES: tuple[str, ...] = (*_QUANTITY_LABELS, NAME_KEY)


def _is_inclined(numbers: Sequence[DimensionlessNumbers]) -> bool:
    """True when any run carries a non-zero gravity inclination."""
    return any(n.inclination_deg is not None and abs(n.inclination_deg) > 0.0 for n in numbers)


def _auto_candidates(numbers: Sequence[DimensionlessNumbers]) -> list[str]:
    """One representative per family, in the order a label renders them."""
    return ["bo_parallel" if _is_inclined(numbers) else "bo", "oh", "re"]


def _varies(values: Sequence[float | None]) -> bool:
    """True when *values* spread by more than :data:`_SIGNIFICANT_SPREAD`.

    A run whose value could not be resolved counts as differing from one whose
    value could, so a partially resolvable quantity still separates the runs.
    """
    resolved = [value for value in values if value is not None]
    if not resolved:
        return False
    if len(resolved) != len(values):
        return True
    scale = max(abs(value) for value in resolved)
    if scale == 0.0:
        return False
    return (max(resolved) - min(resolved)) / scale > _SIGNIFICANT_SPREAD


def resolve_label_keys(
    numbers: Sequence[DimensionlessNumbers],
    requested: Sequence[str] | None = None,
) -> list[str]:
    """Return the label keys to render for a set of runs.

    Args:
        numbers: One :class:`DimensionlessNumbers` per run being compared.
        requested: Explicit keys from ``--label-param``. Taken as given, in the
            order supplied, without the family collapse or the differing-only
            test. ``None`` or empty selects automatically.

    Returns:
        Keys from :data:`LABEL_PARAM_CHOICES`; empty when nothing distinguishes
        the runs, which callers read as "fall back to the run names".
    """
    if requested:
        return [key for key in requested if key in LABEL_PARAM_CHOICES]
    return [key for key in _auto_candidates(numbers) if _varies([getattr(n, key) for n in numbers])]


#: Decimal places every value in a label is rendered with. Fixed rather than
#: significant-figure formatting so the terms line up down the legend; a legend
#: is read by comparing entries, not by reading one to full precision. Values
#: small enough to round to the same text collide, and :func:`_disambiguate`
#: then falls back to naming the runs.
_LABEL_DECIMALS = 2


def _term(key: str, name: str, numbers: DimensionlessNumbers) -> str | None:
    """One rendered label term, or ``None`` when this run cannot supply it."""
    if key == NAME_KEY:
        return name
    value = getattr(numbers, key, None)
    if value is None:
        return None
    return f"${_QUANTITY_LABELS[key]} = {value:.{_LABEL_DECIMALS}f}$"


def _disambiguate(labels: Sequence[str], names: Sequence[str]) -> list[str]:
    """Suffix the run name onto labels shared by more than one run."""
    counts = Counter(labels)
    return [
        f"{label} ({name})" if counts[label] > 1 and label != name else label
        for label, name in zip(labels, names, strict=True)
    ]


def build_run_labels(
    names: Sequence[str],
    numbers: Sequence[DimensionlessNumbers],
    keys: Sequence[str],
) -> list[str]:
    """Render one legend label per run.

    Args:
        names: Fallback label per run (``simulation_name`` or the directory).
        numbers: The matching :class:`DimensionlessNumbers`, same length.
        keys: Label keys from :func:`resolve_label_keys`.

    Returns:
        One label per run. A run supplying none of *keys* keeps its name, so a
        single unresolvable run does not strip the numbers off the others.
    """
    labels: list[str] = []
    for name, dn in zip(names, numbers, strict=True):
        terms = [term for term in (_term(key, name, dn) for key in keys) if term is not None]
        labels.append(", ".join(terms) if terms else name)
    return _disambiguate(labels, names)


def label_sort_key(numbers: DimensionlessNumbers, keys: Sequence[str]) -> tuple[float, ...]:
    """Sort runs by their labelled quantities, so a legend reads monotonically.

    Runs missing a value sort last rather than raising, keeping the ordering
    total for a partially resolvable set.
    """
    values = []
    for key in keys:
        if key == NAME_KEY:
            continue
        value = getattr(numbers, key, None)
        values.append(math.inf if value is None else float(value))
    return tuple(values)
