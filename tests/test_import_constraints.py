"""Import constraints of the core: ``src.protocols`` and the pipeline stay free of plotting."""

from __future__ import annotations
import ast
import subprocess
import sys
from pathlib import Path

#: Third-party packages ``src.protocols`` may import at runtime.
_ALLOWED_THIRD_PARTY = frozenset({"jax", "numpy"})
_PROTOCOLS = Path(__file__).resolve().parents[1] / "src" / "protocols.py"


def _runtime_imports(tree: ast.Module) -> set[str]:
    """Top-level package names imported at module level, outside ``if TYPE_CHECKING:``.

    Imports inside functions and methods run on demand, never on import of the module.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module.split(".")[0])
    return names


def test_protocols_imports_only_the_standard_library_jax_and_numpy():
    tree = ast.parse(_PROTOCOLS.read_text(encoding="utf-8"))

    foreign = _runtime_imports(tree) - sys.stdlib_module_names - _ALLOWED_THIRD_PARTY

    assert foreign == set()


def test_the_pipeline_does_not_import_matplotlib():
    """Runs in a fresh interpreter: the test session has long since imported matplotlib."""
    code = "import sys\nimport src.pipeline.setup\nprint('matplotlib' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=True)  # noqa: S603

    assert result.stdout.strip() == "False"
