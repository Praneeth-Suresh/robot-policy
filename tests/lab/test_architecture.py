"""AST checks for dependency direction and task isolation; no imports are executed."""

import ast
import importlib.util
import re
import sys
import unittest
from pathlib import Path

__all__: list[str] = []

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "guided_streaming_policy/lab"
PREFIX = "guided_streaming_policy.lab"
EDGES = {
    "schema": set(),
    "adapters": {"schema", "tasks.base"},
    "exporter": {"schema", "adapters", "tasks.base"},
    "doctor": {"schema", "tasks.base"},
    "pairing": {"schema", "tasks.base"},
    "checks": {"schema", "tasks.base"},
    "gate": {"schema", "pairing", "checks", "tasks.base"},
    "report": {"schema"},
    "tasks.base": {"schema"},
    "tasks.registry": {"tasks.base"},
    "testing": {"schema", "adapters", "exporter", "doctor", "pairing", "checks", "gate", "report",
                "tasks.base", "tasks.registry"},
    "__init__": set(),
    "tasks": set(),
}
TASK_NAME = re.compile("pusht|push-t", re.IGNORECASE)


def module_name(path: Path) -> str:
    parts = list(path.relative_to(LAB).with_suffix("").parts)
    if parts[-1] == "__init__" and len(parts) > 1:
        parts.pop()
    return ".".join(parts)


def node_name(module: str) -> str:
    parts = module.split(".")
    if parts[0] == "schema":
        return "schema"
    return ".".join(parts[:2]) if parts[0] == "tasks" else parts[0]


def import_targets(item: ast.AST, parent: str) -> list[str]:
    if isinstance(item, ast.Import):
        return [alias.name for alias in item.names]
    if isinstance(item, ast.ImportFrom):
        base = item.module or ""
        if item.level:
            base = importlib.util.resolve_name("." * item.level + base, parent)
        if base in ("guided_streaming_policy", PREFIX, PREFIX + ".tasks", PREFIX + ".schema"):
            return [base + "." + alias.name for alias in item.names]
        return [base]
    if isinstance(item, ast.Call) and item.args:
        name = ast.unparse(item.func)
        target = item.args[0]
        if name in ("importlib.import_module", "import_module") and isinstance(
            target, ast.Constant
        ):
            if isinstance(target.value, str):
                return [target.value]
    return []


def lab_imports(source: str, module: str, *, package: bool = False) -> list[tuple[str, int]]:
    """Resolve absolute/relative imports, including modules imported from packages."""
    result = []
    parent = (
        module
        if module.startswith("guided_streaming_policy.")
        else PREFIX + ("." + module if module != "__init__" else "")
    )
    if not package:
        parent = parent.rpartition(".")[0]
    for item in ast.walk(ast.parse(source)):
        targets = import_targets(item, parent)
        for target in targets:
            if target.startswith(PREFIX + "."):
                result.append((target[len(PREFIX) + 1 :], item.lineno))
            elif target == PREFIX:
                result.append(("__init__", item.lineno))
    return result


def forbidden_edges(source: str, module: str, *, package: bool = False) -> list[str]:
    """Return every forbidden edge with its source line and resolved target."""
    origin = node_name(module)
    failures = []
    for target, line in lab_imports(source, module, package=package):
        destination = node_name(target)
        allowed = EDGES.get(origin, set())
        if origin.startswith("tasks.") and origin not in EDGES:
            allowed = {origin, "schema", "adapters", "tasks.base"}
        if origin != "cli" and destination != origin and destination not in allowed:
            failures.append(f"{module}:{line}: {origin} -> {target}")
    return failures


class ArchitectureTests(unittest.TestCase):
    def test_allowed_import_edges(self) -> None:
        failures = []
        for path in sorted(LAB.rglob("*.py")):
            failures.extend(
                forbidden_edges(
                    path.read_text(), module_name(path), package=path.name == "__init__.py"
                )
            )
        self.assertEqual(failures, [], "Forbidden lab edges:\n" + "\n".join(failures))

    def test_only_wiring_and_same_task_reference_task_implementations(self) -> None:
        failures = []
        for path in sorted(LAB.rglob("*.py")):
            module = module_name(path)
            for target, line in lab_imports(
                path.read_text(), module, package=path.name == "__init__.py"
            ):
                parts = target.split(".")
                if len(parts) < 2 or parts[0] != "tasks" or parts[1] in ("base", "registry"):
                    continue
                if (
                    module not in ("cli", "tasks.registry")
                    and not module.startswith(".".join(parts[:2]) + ".")
                    and module != ".".join(parts[:2])
                ):
                    failures.append(f"{path.relative_to(ROOT)}:{line}: references {target}")
        self.assertEqual(failures, [], "Task implementation leakage:\n" + "\n".join(failures))

    def test_research_never_imports_lab(self) -> None:
        paths = set()
        for directory in (
            "guided_streaming_policy/core",
            "guided_streaming_policy/pusht",
            "Push-T",
            "Robomimic",
        ):
            paths.update((ROOT / directory).rglob("*.py"))
        for name in ("doctor", "paths", "pusht_legacy", "robomimic_legacy"):
            paths.add(ROOT / "guided_streaming_policy" / f"{name}.py")
        failures = []
        for path in sorted(paths):
            if not path.is_file():
                continue
            module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            for target, line in lab_imports(path.read_text(), module):
                failures.append(f"{path.relative_to(ROOT)}:{line}: imports lab.{target}")
        self.assertEqual(failures, [], "Research imports lab:\n" + "\n".join(failures))

    def test_core_has_no_task_names(self) -> None:
        failures = []
        paths = [*LAB.glob("*.py"), *(LAB / "schema").glob("*.py"), *(LAB / "assets").glob("*")]
        for path in sorted(paths):
            source = path.read_text()
            if path.name == "cli.py":
                # Only the three --task defaults may mention the default task.
                source, count = re.subn(
                    r'add_argument\("--task", default="pusht", '
                    r'help="Task adapter name \(default: pusht\)\."\)',
                    "",
                    source,
                )
                self.assertEqual(count, 3, "Review default-task exemptions in cli.py")
            for line, text in enumerate(source.splitlines(), 1):
                if TASK_NAME.search(text):
                    failures.append(f"{path.relative_to(ROOT)}:{line}: {text.strip()}")
        self.assertEqual(failures, [], "Task names in core:\n" + "\n".join(failures))

    def test_default_install_imports_only_standard_library_and_lab(self) -> None:
        failures = []
        for path in sorted(LAB.rglob("*.py")):
            if path.is_relative_to(LAB / "tasks/pusht"):
                continue
            for item in ast.walk(ast.parse(path.read_text())):
                for target in import_targets(item, PREFIX):
                    root = target.partition(".")[0]
                    if root not in sys.stdlib_module_names and not target.startswith(PREFIX):
                        failures.append(f"{path.relative_to(ROOT)}:{item.lineno}: imports {target}")
        self.assertEqual(failures, [], "Default-install dependencies:\n" + "\n".join(failures))

    def test_checker_rejects_synthetic_violations(self) -> None:
        cases = (
            ('importlib.import_module("guided_streaming_policy.lab.gate")', "report", False),
            ("from guided_streaming_policy.lab import gate", "report", False),
            ("import guided_streaming_policy.lab.doctor as d", "schema.traces", False),
            ("from . import doctor", "report", False),
            ("from .. import report", "schema.traces", False),
            ("from ..tasks import pusht", "schema", True),
            ("from guided_streaming_policy.lab.tasks.pusht import adapter", "checks", False),
        )
        for source, module, package in cases:
            with self.subTest(source=source):
                failures = forbidden_edges(source, module, package=package)
                self.assertEqual(len(failures), 1)
                self.assertIn(":1:", failures[0])
        self.assertEqual(forbidden_edges("from .traces import Episode", "schema", package=True), [])
