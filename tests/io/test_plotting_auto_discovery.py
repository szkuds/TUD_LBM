"""Guard the auto-discovery convention in :mod:`src.simulation_io.plotting`.

The package follows the repo-wide rule: :func:`~src.operators._loader.auto_load_operators`
imports every ``_*.py`` and skips every public module. That replaces a
hand-maintained list of side-effect imports, but it introduces a failure mode of
its own -- a *public* module carrying an operator decorator is never scanned, so
its operator silently vanishes from the registry rather than raising.

The exclusion is also load-bearing for import cost: ``regime_map_plot`` registers
nothing and pulls in scipy, which would roughly double this package's import time.
Keeping it public excludes it by the same rule that performs the discovery.
"""

from __future__ import annotations
import ast
import pathlib
import subprocess
import sys

_PLOTTING_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "simulation_io" / "plotting"

#: Decorators that place an operator in the registry.
_REGISTRATION_DECORATORS = frozenset({"plotting_operator", "analysis_operator"})

#: Private, but shared helpers rather than an operator module. Being scanned is a
#: no-op -- the operator modules import it anyway.
_PRIVATE_NON_OPERATOR_MODULES = frozenset({"_analysis_common"})


def _decorator_names(tree: ast.Module) -> set[str]:
    """Return the bare names of every decorator applied anywhere in ``tree``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef | ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def _registering_modules() -> dict[str, bool]:
    """Map each module stem in the package to whether it registers an operator."""
    return {
        path.stem: bool(_decorator_names(ast.parse(path.read_text())) & _REGISTRATION_DECORATORS)
        for path in sorted(_PLOTTING_DIR.glob("*.py"))
        if path.stem != "__init__"
    }


def test_no_public_module_registers_an_operator() -> None:
    """A public module is never scanned, so registering from one drops the operator silently."""
    offenders = sorted(
        stem for stem, registers in _registering_modules().items() if registers and not stem.startswith("_")
    )
    assert not offenders, (
        f"public module(s) {offenders} register an operator but are skipped by auto-discovery; rename to _<name>.py"
    )


def test_every_private_module_is_an_operator_module() -> None:
    """Keeps the ``_*.py`` prefix meaningful, so the convention stays readable."""
    stray = sorted(
        stem
        for stem, registers in _registering_modules().items()
        if stem.startswith("_") and not registers and stem not in _PRIVATE_NON_OPERATOR_MODULES
    )
    assert not stray, f"private module(s) {stray} register nothing; make them public or add to the documented exception"


def _probe(expression: str) -> str:
    """Evaluate *expression* in a fresh interpreter that has imported the package.

    A subprocess is not incidental here. The registry is process-global and
    pytest shares it across the whole session, so sibling test modules that
    import ``plotting._force`` &c. directly leave every operator registered no
    matter what the package itself did. Only a clean interpreter observes what a
    bare ``import src.simulation_io.plotting`` -- which is all the CLI does --
    actually registers.
    """
    probe = f"import sys, src.simulation_io.plotting; print({expression})"
    # argv is this interpreter plus a literal probe: no untrusted input.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=_PLOTTING_DIR.parents[2].parent,
    )
    return result.stdout.strip()


def test_package_import_alone_registers_every_operator() -> None:
    """Importing the package must register both kinds, with no other import needed.

    Guards the ``auto_load_operators`` call in ``__init__``: dropping it leaves
    only the operators the explicit re-exports happen to pull in, which silently
    shortens the ``visualise`` field menu while the test suite stays green.
    """
    counts = _probe(
        "(len(__import__('src.registry', fromlist=['x']).get_operator_names('plotting')), "
        "len(__import__('src.registry', fromlist=['x']).get_operator_names('analysis')))"
    )
    assert counts == "(8, 16)", (
        f"bare package import registered {counts}, expected (8, 16) -- is auto_load_operators still called?"
    )


def test_importing_the_package_does_not_pull_in_scipy() -> None:
    """``regime_map_plot`` stays out of package init -- the reason it is public."""
    observed = _probe("('scipy' in sys.modules, 'src.simulation_io.plotting.regime_map_plot' in sys.modules)")
    assert observed == "(False, False)", f"unexpected package-init imports: {observed}"
