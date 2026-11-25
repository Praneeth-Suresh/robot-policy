"""Policy Trace Lab command registration and output."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from typing import Any, NoReturn

from guided_streaming_policy.lab.adapters import format_policy_rows, load_pins
from guided_streaming_policy.lab.tasks.registry import get_task

__all__ = ["LabArgumentParser", "add_lab_parser", "run_lab"]

_EXIT_OK = 0
_EXIT_INVALID_INPUT = 2
_EXIT_USAGE_OR_INTERNAL = 3
_LAB_HELP = "Policy Trace Lab: validate robot data, compare policies, replay regressions."


class LabArgumentParser(argparse.ArgumentParser):
    """Keep lab usage failures at exit code 3 without changing legacy commands."""

    def error(self, message: str) -> NoReturn:
        """Print usage and a diagnostic to stderr, then exit with usage code 3."""
        self.print_usage(sys.stderr)
        self.exit(_EXIT_USAGE_OR_INTERNAL, f"{self.prog}: error: {message}\n")

    def parse_known_args(
        self, args: Iterable[str] | None = None, namespace: Any = None
    ) -> tuple[Any, list[str]]:
        """Reject leftover lab arguments here so the legacy root cannot exit 2."""
        parsed, remaining = super().parse_known_args(args, namespace)
        if remaining:
            self.error(f"unrecognized arguments: {' '.join(remaining)}")
        return parsed, remaining


def add_lab_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the Policy Trace Lab group with its own usage-error convention."""
    # argparse selects parser classes on the group, rather than on add_parser.
    parser_class = subparsers._parser_class
    subparsers._parser_class = LabArgumentParser
    try:
        parser = subparsers.add_parser("lab", help=_LAB_HELP, description=_LAB_HELP)
    finally:
        subparsers._parser_class = parser_class
    commands = parser.add_subparsers(
        dest="lab_command", parser_class=LabArgumentParser, required=True
    )
    policies = commands.add_parser(
        "policies",
        help="List Policy Trace Lab reference policies.",
        description="Policy Trace Lab reference policy inventory.",
        epilog="Exit codes: 0 listed, 3 usage error.",
    )
    policies.add_argument("--task", default="pusht", help="Task adapter name (default: pusht).")


def run_lab(args: argparse.Namespace) -> int:
    """Dispatch a lab command and translate input/usage/internal failures to exit codes."""
    if args.lab_command != "policies":
        sys.stderr.write(f"Unknown Policy Trace Lab command: {args.lab_command}\n")
        return _EXIT_USAGE_OR_INTERNAL
    try:
        adapter = get_task(args.task)
    except KeyError as error:
        sys.stderr.write(f"{error}\n")
        return _EXIT_USAGE_OR_INTERNAL
    try:
        pins = load_pins(adapter.pins_path())["tasks"][adapter.name]
        rows = format_policy_rows(adapter.policy_statuses(), pins)
        sys.stdout.write("\n".join(rows) + "\n")
    except (OSError, ValueError, KeyError) as error:
        sys.stderr.write(f"Cannot read Policy Trace Lab input: {error}\n")
        return _EXIT_INVALID_INPUT
    except Exception as error:
        sys.stderr.write(f"Policy Trace Lab failed: {error}\n")
        return _EXIT_USAGE_OR_INTERNAL
    return _EXIT_OK
