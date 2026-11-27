"""The `gsp lab tasks` command group: list task packs, scaffold one, check one.

Exit codes: list 0 or 3; new 0, 2 (refused) or 3; check 0 (all pass), 1 (a failure),
2 (the pack cannot be loaded) or 3.
"""

from __future__ import annotations

import argparse
import importlib.resources
import importlib.util
import re
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import testing
from guided_streaming_policy.lab.tasks import registry
from guided_streaming_policy.lab.tasks.base import TaskAdapter

__all__ = ["ScaffoldError", "add_tasks_parser", "run_tasks", "scaffold_pack"]

_EXIT_OK, _EXIT_FAILED, _EXIT_REFUSED, _EXIT_USAGE_OR_INTERNAL = 0, 1, 2, 3
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,29}$")
_LAB_PACKAGE = "guided_streaming_policy.lab"
_BUILD_TIMEOUT_SECONDS = 300
_CASES_FILE = "cases.json"
_BUILD_ERROR_CHARACTERS = 400
_TEMPLATES = (
    ("pyproject.toml.tmpl", "pyproject.toml"),
    ("README.md.tmpl", "README.md"),
    ("__init__.py.tmpl", "{module}/__init__.py"),
    ("simulator.py.tmpl", "{module}/simulator.py"),
    ("adapter.py.tmpl", "{module}/adapter.py"),
    ("build_fixtures.py.tmpl", "tools/build_fixtures.py"),
    ("test_contract.py.tmpl", "tests/test_contract.py"),
    ("comparison.json.tmpl", "configs/comparison.json"),
    ("gate.json.tmpl", "configs/gate.json"),
)
_LIST_EXITS = "Exit codes: 0 listed, 3 usage error."
_NEW_EXITS = ("Exit codes: 0 written, 2 refused (name taken or invalid, or output directory not "
              "empty), 3 usage or internal error.")
_CHECK_EXITS = ("Exit codes: 0 all checks pass, 1 a check failed, 2 the pack cannot be loaded, "
                "3 usage error.")


class ScaffoldError(Exception):
    """The requested pack cannot be written; the message says why."""


def add_tasks_parser(commands: argparse._SubParsersAction[Any]) -> None:
    """Register `gsp lab tasks list|new|check` under the lab command group."""
    parser = commands.add_parser(
        "tasks", help="List, scaffold and check Policy Trace Lab task packs.",
        description="Policy Trace Lab task packs: one simulator, its conditions and its "
                    "reference policies, discovered through the gsp_lab.tasks entry point.",
        epilog="Exit codes: see each subcommand.")
    group = parser.add_subparsers(dest="tasks_command", parser_class=type(parser), required=True)
    listing = group.add_parser(
        "list", help="List installed task packs with their status.",
        description="Policy Trace Lab lists every discoverable task pack. Each pack is loaded "
                    "to read its contract version, policies and conditions; one that fails to "
                    "load is shown as broken and does not affect the others.",
        epilog=_LIST_EXITS)
    listing.add_argument("--verbose", action="store_true",
                         help="Also list each condition with its description.")
    new = group.add_parser(
        "new", help="Write a standalone, pip-installable task pack.",
        description="Policy Trace Lab writes a task pack for a one-dimensional rail simulator "
                    "that really runs, with packaged fixtures, configs and a contract test.",
        epilog=_NEW_EXITS)
    new.add_argument("name", help="Task name: lowercase letters, digits, '-' or '_'.")
    new.add_argument("--out", type=Path, required=True,
                     help="Output directory; must be empty or absent.")
    check = group.add_parser(
        "check", help="Run the conformance checks against one task pack.",
        description="Policy Trace Lab runs the conformance suite and prints pass, fail or "
                    "skip, with a reason, for each check.",
        epilog=_CHECK_EXITS)
    check.add_argument("name", help="Task name, as shown by `gsp lab tasks list`.")


def _pack_status(adapter: TaskAdapter) -> str:
    """`ready` means the pack loads and ships its pin file and fixture declarations.

    Whether the simulator and checkpoints are installed is per policy: see
    `gsp lab policies`.
    """
    missing = []
    if not adapter.pins_path().is_file():
        missing.append("pin file")
    if not (adapter.fixtures().root / _CASES_FILE).is_file():
        missing.append(f"fixtures/{_CASES_FILE}")
    return f"missing assets: {', '.join(missing)}" if missing else "ready"


def _describe(entry: registry.TaskEntry) -> tuple[str, str, str, TaskAdapter | None]:
    """Return contract version, status and policy count, loading the pack to find out."""
    try:
        adapter = registry.get_task(entry.name)
        return (str(adapter.contract_version), _pack_status(adapter),
                str(len(adapter.policies())), adapter)
    except registry.TaskLoadError as error:
        return "-", f"broken: {error}", "-", None
    except Exception as error:  # a pack's own code may fail in any way while being inspected
        return "-", f"broken: {type(error).__name__}: {error}", "-", None


def _row(values: tuple[str, ...], widths: tuple[int, ...]) -> str:
    cells = (value.ljust(width) for value, width in zip(values, widths, strict=True))
    return "  ".join(cells).rstrip()


def _list(args: argparse.Namespace) -> int:
    header = ("NAME", "DISTRIBUTION", "CONTRACT", "STATUS", "POLICIES", "CONDITIONS")
    rows: list[tuple[str, ...]] = []
    adapters_by_row: list[TaskAdapter | None] = []
    for entry in registry.entries():
        contract, status, count, adapter = _describe(entry)
        names = ", ".join(spec.name for spec in adapter.conditions()) if adapter else "-"
        rows.append((entry.name, f"{entry.distribution} {entry.version}", contract, status,
                     count, names))
        adapters_by_row.append(adapter)
    widths = tuple(max(len(line[column]) for line in [header, *rows])
                   for column in range(len(header)))
    lines = [_row(header, widths)]
    for row, adapter in zip(rows, adapters_by_row, strict=True):
        lines.append(_row(row, widths))
        if args.verbose and adapter is not None:
            lines.extend(f"    {spec.name}: {spec.description}"
                         + ("" if spec.runnable else " (declared; not runnable)")
                         for spec in adapter.conditions())
    sys.stdout.write("\n".join(lines) + "\n")
    return _EXIT_OK


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[-_]", name) if part)


def _validate_scaffold(name: str, out: Path) -> str:
    if not _NAME.match(name):
        raise ScaffoldError(f"invalid task name {name!r}: use 1 to 30 lowercase letters, "
                            "digits, '-' or '_', starting with a letter")
    if name in registry.task_names():
        raise ScaffoldError(f"task name {name!r} is already registered")
    module = name.replace("-", "_")
    if importlib.util.find_spec(module) is not None:
        raise ScaffoldError(f"{module!r} is already an importable Python module; pick another")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ScaffoldError(f"output directory {out} is not empty; refusing to overwrite")
    return module


def scaffold_pack(name: str, out: Path) -> Path:
    """Write a pack for `name` into `out` and build its fixtures; raise ScaffoldError if refused."""
    out = out.resolve()
    module = _validate_scaffold(name, out)
    values = {"name": name, "module": module, "dist": name.replace("_", "-"),
              "Class": _camel(name)}
    source = importlib.resources.files(_LAB_PACKAGE) / "scaffold"
    for template, target in _TEMPLATES:
        path = out / target.format(module=module)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (source / template).read_text(encoding="utf-8")
        path.write_text(string.Template(text).substitute(values), encoding="utf-8")
    done = subprocess.run([sys.executable, str(out / "tools/build_fixtures.py")], cwd=out,
                          capture_output=True, text=True, timeout=_BUILD_TIMEOUT_SECONDS)
    if done.returncode != 0:
        raise RuntimeError("building the fixtures failed: "
                           + done.stderr.strip()[-_BUILD_ERROR_CHARACTERS:])
    return out


def _new(args: argparse.Namespace) -> int:
    try:
        out = scaffold_pack(args.name, args.out)
    except ScaffoldError as error:
        sys.stderr.write(f"Cannot write the task pack: {error}\n")
        return _EXIT_REFUSED
    sys.stdout.write(
        f"task pack {args.name}: {out}\n"
        f"next: cd {out} && pip install . && gsp lab tasks check {args.name}\n")
    return _EXIT_OK


def _check(args: argparse.Namespace) -> int:
    try:
        adapter = registry.get_task(args.name)
    except (KeyError, registry.TaskLoadError) as error:
        sys.stderr.write(f"Cannot load task {args.name!r}: {error}\n")
        return _EXIT_REFUSED
    results = testing.run_conformance(adapter)
    for result in results:
        sys.stdout.write(f"{result.status:<4} {result.name}: {result.detail}".rstrip() + "\n")
    counts = {status: sum(r.status == status for r in results)
              for status in (testing.PASS, testing.FAIL, testing.SKIP)}
    sys.stdout.write(f"{counts[testing.PASS]} passed, {counts[testing.FAIL]} failed, "
                     f"{counts[testing.SKIP]} skipped\n")
    return _EXIT_FAILED if counts[testing.FAIL] else _EXIT_OK


def run_tasks(args: argparse.Namespace) -> int:
    """Dispatch a `gsp lab tasks` subcommand."""
    handlers = {"list": _list, "new": _new, "check": _check}
    try:
        return handlers[args.tasks_command](args)
    except Exception as error:
        sys.stderr.write(f"Policy Trace Lab failed: {error}\n")
        return _EXIT_USAGE_OR_INTERNAL
