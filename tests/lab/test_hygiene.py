"""Source hygiene, bounded functions, exercised artifacts and the offline CI contract."""

import ast
import fnmatch
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from guided_streaming_policy.lab import schema
from tests.lab.test_architecture import LAB, PREFIX, ROOT, lab_imports, module_name

__all__: list[str] = []

MAX_MODULE_LINES = 400
MAX_FUNCTION_LINES = 50
MAX_COMPLEXITY = 10
MARKERS = re.compile(r"TO" r"DO|FIX" r"ME|XX" r"X")
TRACKED_TASK = re.compile(r"TO" r"DO\(T\d+\)")
SCRATCH_PATTERNS = ("*.orig", "*.bak", "*.rej", "*~", "*.ipynb", "scratch*", "tmp*")
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def complexity(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Approximate C901: decisions + 1, excluding nested function/class bodies.

    Count if/elif, loops, exception handlers, boolean joins, comprehension ifs
    and match cases. With statements do not add a branch (with-free counting).
    """
    score = 1
    pending = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (*FUNCTIONS, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler)):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            score += len(node.ifs)
        elif isinstance(node, ast.match_case):
            score += 1
        pending.extend(ast.iter_child_nodes(node))
    return score


def bound_names(tree: ast.Module) -> set[str]:
    names = set()
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(item.name)
        elif isinstance(item, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in item.names)
        elif isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            names.update(
                node.id
                for target in targets
                for node in ast.walk(target)
                if isinstance(node, ast.Name)
            )
    return names


def export_errors(tree: ast.Module) -> list[str]:
    declarations = [
        item
        for item in tree.body
        if isinstance(item, (ast.Assign, ast.AnnAssign))
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in (item.targets if isinstance(item, ast.Assign) else [item.target])
        )
    ]
    if len(declarations) != 1:
        return ["expected one literal __all__ declaration"]
    value = declarations[0].value
    if not isinstance(value, (ast.List, ast.Tuple)) or not all(
        isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts
    ):
        return ["__all__ must be a list/tuple of string literals"]
    return [
        f"__all__ names missing binding: {item.value}"
        for item in value.elts
        if isinstance(item, ast.Constant) and item.value not in bound_names(tree)
    ]


def annotation_errors(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = function.args
    parameters = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    parameters += [item for item in (args.vararg, args.kwarg) if item is not None]
    missing = [
        arg.arg for arg in parameters if arg.arg not in ("self", "cls") and arg.annotation is None
    ]
    if function.returns is None:
        missing.append("return")
    return missing


def attribute_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return attribute_path(node.value) + "." + node.attr
    return ""


def is_lab_module(candidate: str) -> bool:
    if candidate == PREFIX:
        return True
    path = LAB.joinpath(*candidate[len(PREFIX) + 1 :].split("."))
    return path.is_dir() or path.with_suffix(".py").is_file()


def private_accesses(source: str, module: str, *, package: bool = False) -> list[str]:
    tree = ast.parse(source)
    aliases = {}
    failures = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                {
                    alias.asname or alias.name: alias.name
                    for alias in node.names
                    if alias.name.startswith(PREFIX)
                }
            )
        elif isinstance(node, ast.ImportFrom):
            text = ast.get_source_segment(source, node)
            targets = lab_imports(text or "", module, package=package)
            for target, _line in targets:
                for alias in node.names:
                    full = PREFIX + "." + target
                    if alias.name.startswith("_") and target != module:
                        failures.append(f"{node.lineno}: imports private {full}.{alias.name}")
                    candidate = full if full.endswith("." + alias.name) else full + "." + alias.name
                    if full == PREFIX + ".__init__":
                        candidate = PREFIX
                    if is_lab_module(candidate):
                        aliases[alias.asname or alias.name] = candidate
    return failures + private_attributes(tree, aliases, module)


def private_attributes(tree: ast.Module, aliases: dict[str, str], module: str) -> list[str]:
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not node.attr.startswith("_"):
            continue
        owner = attribute_path(node.value)
        root, separator, rest = owner.partition(".")
        target = aliases.get(owner)
        if target is None and root in aliases:
            target = aliases[root] + (separator + rest if separator else "")
        if target is not None and is_lab_module(target) and target != PREFIX + "." + module:
            failures.append(f"{node.lineno}: accesses private {target}.{node.attr}")
    return failures


class HygieneTests(unittest.TestCase):
    def test_exports_exist_and_are_literal(self) -> None:
        failures = [
            f"{path.relative_to(ROOT)}: {error}"
            for path in sorted(LAB.rglob("*.py"))
            for error in export_errors(ast.parse(path.read_text()))
        ]
        self.assertEqual(failures, [], "Invalid exports:\n" + "\n".join(failures))

    def test_module_line_limit(self) -> None:
        failures = [
            f"{path.relative_to(ROOT)}: {len(path.read_text().splitlines())} lines"
            for path in sorted(LAB.rglob("*.py"))
            if len(path.read_text().splitlines()) > MAX_MODULE_LINES
        ]
        self.assertEqual(failures, [], "Oversized modules:\n" + "\n".join(failures))

    def test_function_size_and_complexity(self) -> None:
        failures = []
        for path in sorted(LAB.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, FUNCTIONS):
                    length = (node.end_lineno or node.lineno) - node.lineno + 1
                    branches = complexity(node)
                    if length > MAX_FUNCTION_LINES or branches > MAX_COMPLEXITY:
                        failures.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: {node.name}: "
                            f"{length} lines, complexity {branches}"
                        )
        self.assertEqual(failures, [], "Function limits exceeded:\n" + "\n".join(failures))

    def test_public_functions_are_annotated(self) -> None:
        failures = []
        for path in sorted(LAB.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, FUNCTIONS) and not node.name.startswith("_"):
                    missing = annotation_errors(node)
                    if missing:
                        failures.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: {node.name}: "
                            + ", ".join(missing)
                        )
        self.assertEqual(failures, [], "Missing annotations:\n" + "\n".join(failures))

    def test_no_cross_module_private_access(self) -> None:
        failures = []
        for path in sorted(LAB.rglob("*.py")):
            failures.extend(
                f"{path.relative_to(ROOT)}:{error}"
                for error in private_accesses(
                    path.read_text(), module_name(path), package=path.name == "__init__.py"
                )
            )
        self.assertEqual(failures, [], "Private lab access:\n" + "\n".join(failures))

    def test_no_print_outside_cli(self) -> None:
        failures = [
            f"{path.relative_to(ROOT)}:{node.lineno}: print()"
            for path in sorted(LAB.rglob("*.py"))
            if path.name != "cli.py"
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        self.assertEqual(failures, [], "Debug output:\n" + "\n".join(failures))

    def test_no_untracked_work_markers(self) -> None:
        failures = []
        for directory in (LAB, ROOT / "tests/lab", ROOT / "docs/lab"):
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                source = TRACKED_TASK.sub("", path.read_text(encoding="utf-8"))
                failures.extend(
                    f"{path.relative_to(ROOT)}:{line}: {match.group()}"
                    for line, text in enumerate(source.splitlines(), 1)
                    for match in MARKERS.finditer(text)
                )
        self.assertEqual(failures, [], "Untracked work markers:\n" + "\n".join(failures))

    @unittest.skipUnless(shutil.which("git"), "git unavailable; cannot inspect tracked files")
    def test_no_tracked_scratch_files(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        failures = [
            name
            for name in result.stdout.split("\0")
            if name
            and any(fnmatch.fnmatch(Path(name).name, pattern) for pattern in SCRATCH_PATTERNS)
        ]
        self.assertEqual(failures, [], "Tracked scratch files: " + ", ".join(failures))

    def test_schema_directory_has_no_extra_files(self) -> None:
        actual = {
            path.relative_to(ROOT / "schemas").as_posix()
            for path in (ROOT / "schemas").rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual, set(schema.generated_schemas()))

    def test_every_schema_has_a_consumer_test(self) -> None:
        # Exclude this guard and the drift method, retaining real schema validations.
        consumers = []
        for path in (ROOT / "tests/lab").glob("test_*.py"):
            if path == Path(__file__):
                continue
            source = path.read_text()
            lines = source.splitlines(keepends=True)
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "test_generated_files_match_builders"
                ):
                    lines[node.lineno - 1 : node.end_lineno] = [""] * (
                        (node.end_lineno or node.lineno) - node.lineno + 1
                    )
            consumers.append("".join(lines))
        source = "\n".join(consumers)
        missing = [
            path.name for path in (ROOT / "schemas").glob("*.json") if path.name not in source
        ]
        self.assertEqual(missing, [], "Schemas without consumer tests: " + ", ".join(missing))

    def test_every_machine_config_is_used(self) -> None:
        source = "\n".join(
            path.read_text()
            for directory in (ROOT / "tests/lab", LAB)
            for path in directory.rglob("*.py")
            if path != Path(__file__)
        )
        configs = sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "lab").rglob("*.json")
            if path.name != "pins.json"
        )
        missing = [path for path in configs if path not in source]
        self.assertEqual(missing, [], "Unused machine configs: " + ", ".join(missing))

    def test_ci_offline_job_contract(self) -> None:
        workflow = ROOT / ".github/workflows/lab.yml"
        self.assertTrue(workflow.is_file(), "Missing offline lab CI workflow")
        source = workflow.read_text()
        commands = (
            "push:",
            "pull_request:",
            "lab-offline:",
            "runs-on: ubuntu-latest",
            "actions/checkout@v4",
            "actions/setup-python@v5",
            "python-version: '3.10'",
            'python -m pip install -e ".[dev]" numpy==1.26.4 zarr==2.18.3',
            "python -m ruff check guided_streaming_policy/lab tests/lab",
            "python -m mypy",
            "python -m vulture",
            "python -m compileall -q guided_streaming_policy Push-T tests",
            "python -m unittest discover -s tests -v",
            "gsp lab demo --out out/demo",
            'test "$demo_code" -eq 1',
            "out/demo/gate/gate.json",
            "REGRESSION",
            "exit_code",
            "separation_proxy",
            "out/demo/report/report.html",
            "Synthetic fixture data. Not a benchmark.",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIn(command, source)

    def test_hygiene_checkers_reject_synthetic_violations(self) -> None:
        self.assertTrue(export_errors(ast.parse('__all__ = ["missing"]')))
        self.assertTrue(export_errors(ast.parse("__all__ = list()")))
        function = ast.parse("def f(x, /, *args, y, **kwargs):\n    pass").body[0]
        self.assertIsInstance(function, ast.FunctionDef)
        self.assertEqual(annotation_errors(function), ["x", "y", "args", "kwargs", "return"])
        branching = ast.parse("def f():\n    if a and b:\n        pass").body[0]
        self.assertEqual(complexity(branching), 3)
        self.assertTrue(private_accesses("from . import schema as s\ns._hidden()", "checks"))
        self.assertTrue(private_accesses("from .schema import _hidden", "checks"))
        self.assertTrue(private_accesses(
            "from guided_streaming_policy import lab as l\nl.schema._hidden()", "checks"))
        self.assertTrue(private_accesses(
            "import guided_streaming_policy.lab.schema as s\ns._hidden()", "checks"))
        self.assertEqual(private_accesses("from . import checks as c\nc._own()", "checks"), [])
