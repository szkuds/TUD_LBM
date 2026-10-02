"""Click option shapes shared by more than one ``run`` flag."""

from __future__ import annotations
from typing import TYPE_CHECKING
import click

if TYPE_CHECKING:
    from collections.abc import Callable
    from click.decorators import FC


def optional_int_option(flag: str, *, default: int, metavar: str, help_text: str) -> Callable[[FC], FC]:
    """A flag that carries an optional positive integer: ``--flag`` or ``--flag N``.

    Absent, the parameter is ``None``; bare, it is *default*; with a value, that
    value. The value is optional, so the flag consumes a following positional;
    the help text gains the default and that placement rule here, once.
    """
    value_name = metavar.strip("[]")
    return click.option(
        flag,
        type=click.IntRange(min=1),
        is_flag=False,
        flag_value=default,
        default=None,
        metavar=metavar,
        help=f"{help_text} {value_name} defaults to {default}. Place after CONFIG_PATH, or write {flag}={value_name}",
    )


def optional_int(value: object) -> int | None:
    """Narrow an :func:`optional_int_option` value: ``None`` when the flag was absent."""
    return value if isinstance(value, int) else None
