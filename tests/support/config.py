"""Building test configs from a base with overrides."""

from __future__ import annotations
from typing import Any


def merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply *overrides* to *base* in place, section by section.

    ``multiphase`` and ``forces`` overrides merge into the base's section, so a
    test names only what it changes. A new gravity force replaces the base's,
    because a run may have only one, and an empty ``forces`` removes them all.
    """
    merged = dict(overrides)
    if base.get("multiphase") and merged.get("multiphase") is not None:
        merged["multiphase"] = {**base["multiphase"], **merged["multiphase"]}
    if merged.get("forces"):
        kept = dict(base.get("forces") or {})
        if any(name.startswith("gravity") for name in merged["forces"]):
            kept = {name: params for name, params in kept.items() if not name.startswith("gravity")}
        merged["forces"] = {**kept, **merged["forces"]}
    base.update(merged)
