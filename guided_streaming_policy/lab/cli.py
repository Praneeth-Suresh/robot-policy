"""Policy Trace Lab command registration and output."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, NoReturn

from guided_streaming_policy.lab import adapters, doctor, exporter, schema
from guided_streaming_policy.lab.adapters import format_policy_rows, load_pins
from guided_streaming_policy.lab.tasks.base import TaskAdapter
from guided_streaming_policy.lab.tasks.registry import get_task

__all__ = ["LabArgumentParser", "add_lab_parser", "run_lab"]

_EXIT_OK = 0
_EXIT_INVALID_INPUT = 2
_EXIT_USAGE_OR_INTERNAL = 3
_LAB_HELP = "Policy Trace Lab: validate robot data, compare policies, replay regressions."
_DOCTOR_EXITS = (
    "Exit codes: 0 clean, 1 errors found, 2 input unreadable, 3 usage or internal error."
)


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
        parser = subparsers.add_parser(
            "lab", help=_LAB_HELP, description=_LAB_HELP, epilog=_DOCTOR_EXITS
        )
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
    _add_doctor_parser(commands)
    _add_run_parser(commands)


def _add_doctor_parser(commands: argparse._SubParsersAction[LabArgumentParser]) -> None:
    parser = commands.add_parser(
        "doctor", help="Validate Policy Trace Lab traces or native demonstrations.",
        description="Policy Trace Lab Doctor accepts an episode trace directory, a run directory "
                    "containing episode traces, or a native demonstration file/directory.",
        epilog=_DOCTOR_EXITS,
    )
    parser.add_argument("input", type=Path, help="Trace directory or native demonstration input.")
    parser.add_argument("--task", default="pusht", help="Task adapter name (default: pusht).")
    parser.add_argument("--out", type=Path, default=Path("out/doctor"),
                        help="Report directory (default: out/doctor). Preserves earlier outputs.")


def _add_run_parser(commands: argparse._SubParsersAction[LabArgumentParser]) -> None:
    parser = commands.add_parser(
        "run", help="Run the simulator and record Policy Trace Lab episodes.",
        description="Policy Trace Lab runs the simulator for the comparison's explicit "
                    "conditions and seeds. Requires the task's simulator packages and assets. "
                    "Push-T assets: data/pusht_cchi_v7_replay.zarr.zip; checkpoints/"
                    "pusht_state_1000ep_si.ckpt, CCG_V6_ep200.pth (critic), "
                    "pusht_state_100ep_dp.ckpt, pusht_state_flow_100ep.ckpt. "
                    "Only the selected policy's checkpoints are required.",
        epilog="Exit codes: 0 all trials recorded (including logged crashes), "
               "1 findings/regression, 2 invalid/unreadable/incomplete input, "
               "3 usage or internal error.",
    )
    parser.add_argument("--comparison", type=Path, required=True, help="Comparison JSON file.")
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--out", type=Path, help="Output directory "
                        "(default: out/runs/<comparison name>/<variant>); must be empty.")


def _log_trial(message: str) -> None:
    sys.stderr.write(message + "\n")


def _run_export(args: argparse.Namespace) -> int:
    config = exporter.load_comparison(args.comparison)
    try:
        adapter = get_task(config.task)
    except KeyError as error:
        raise exporter.SetupError(str(error)) from error
    out = args.out if args.out is not None else exporter.default_out_dir(
        args.comparison, args.variant)
    command = f"gsp lab run --comparison {args.comparison} --variant {args.variant} --out {out}"
    with redirect_stdout(sys.stderr):
        exporter.export_variant(adapter, args.comparison, args.variant, out, command=command,
                                log=_log_trial)
    sys.stdout.write(f"run: {out} ({len(config.conditions) * len(config.seeds)} trials)\n")
    return _EXIT_OK


def _run_policies(adapter: TaskAdapter) -> int:
    try:
        pins = load_pins(adapter.pins_path())["tasks"][adapter.name]
        rows = format_policy_rows(adapter.policy_statuses(), pins)
        sys.stdout.write("\n".join(rows) + "\n")
    except (OSError, ValueError, KeyError) as error:
        sys.stderr.write(f"Cannot read Policy Trace Lab input: {error}\n")
        return _EXIT_INVALID_INPUT
    return _EXIT_OK


def _run_doctor(args: argparse.Namespace, adapter: TaskAdapter) -> int:
    path: Path = args.input
    if not path.exists():
        raise schema.TraceReadError(f"Input does not exist: {path}")
    episodes = (adapters.read_traces(path) if adapters.is_trace_dir(path)
                else adapter.read_demonstrations(path))
    report = doctor.check_episodes(episodes, input_path=str(path),
                                   input_checksum=adapters.path_checksum(path))
    output = args.out / "doctor.json"
    adapters.write_report_file(output, schema.canonical_json(report))
    sys.stdout.write(doctor.render_summary(report) + f"\ndoctor.json: {output}\n")
    return doctor.exit_code(report)


def run_lab(args: argparse.Namespace) -> int:
    """Dispatch a lab command and translate input/usage/internal failures to exit codes."""
    if args.lab_command not in ("policies", "doctor", "run"):
        sys.stderr.write(f"Unknown Policy Trace Lab command: {args.lab_command}\n")
        return _EXIT_USAGE_OR_INTERNAL
    try:
        if args.lab_command == "run":
            return _run_export(args)
        try:
            adapter = get_task(args.task)
        except KeyError as error:
            sys.stderr.write(f"{error}\n")
            return _EXIT_USAGE_OR_INTERNAL
        if args.lab_command == "doctor":
            return _run_doctor(args, adapter)
        return _run_policies(adapter)
    except (OSError, schema.TraceReadError, exporter.SetupError) as error:
        sys.stderr.write(f"Cannot read or write Policy Trace Lab input/output: {error}\n")
        return _EXIT_INVALID_INPUT
    except Exception as error:
        sys.stderr.write(f"Policy Trace Lab failed: {error}\n")
        return _EXIT_USAGE_OR_INTERNAL
