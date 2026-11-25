"""Policy Trace Lab command registration and output."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, NoReturn

from guided_streaming_policy.lab import adapters, doctor, exporter, gate, pairing, report, schema
from guided_streaming_policy.lab.adapters import format_policy_rows, load_pins
from guided_streaming_policy.lab.tasks import registry
from guided_streaming_policy.lab.tasks.base import FixtureSet, TaskAdapter
from guided_streaming_policy.lab.tasks.registry import get_task

__all__ = ["LabArgumentParser", "add_lab_parser", "run_lab"]

_EXIT_OK = 0
_EXIT_INVALID_INPUT = 2
_EXIT_USAGE_OR_INTERNAL = 3
_LAB_HELP = "Policy Trace Lab: validate robot data, compare policies, replay regressions."
_FIXTURES_MISSING = "fixtures not found; run from a source checkout of the repository"
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
    _add_run_parser(commands, policies.get_default("task"))
    _add_compare_parser(commands)
    _add_gate_parser(commands)
    _add_report_parser(commands)
    _add_demo_parser(commands)


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


def _add_run_parser(commands: argparse._SubParsersAction[LabArgumentParser], task: str) -> None:
    assets = ", ".join(sorted(registry.get_task(task).probe().assets))
    parser = commands.add_parser(
        "run", help="Run the simulator and record Policy Trace Lab episodes.",
        description="Policy Trace Lab runs the simulator for the comparison's explicit "
                    "conditions and seeds. Requires the task's simulator packages and assets. "
                    f"Default task assets: {assets}. "
                    "Only the selected policy's checkpoints are required.",
        epilog="Exit codes: 0 all trials recorded (including logged crashes), "
               "1 findings/regression, 2 invalid/unreadable/incomplete input, "
               "3 usage or internal error.",
    )
    parser.add_argument("--comparison", type=Path, required=True, help="Comparison JSON file.")
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--out", type=Path, help="Output directory "
                        "(default: out/runs/<comparison name>/<variant>); must be empty.")


def _add_compare_parser(commands: argparse._SubParsersAction[LabArgumentParser]) -> None:
    parser = commands.add_parser(
        "compare", help="Compare paired Policy Trace Lab runs.",
        description="Policy Trace Lab compares recorded baseline and candidate runs "
                    "against an explicit condition/seed matrix.",
        epilog="Exit codes: 0 valid comparison, 1 findings/regression (reserved), "
               "2 invalid/unreadable/incomplete input, 3 usage or internal error.",
    )
    parser.add_argument("baseline_run", type=Path, help="Baseline run directory.")
    parser.add_argument("candidate_run", type=Path, help="Candidate run directory.")
    parser.add_argument("--comparison", type=Path, required=True, help="Comparison JSON file.")
    parser.add_argument("--out", type=Path, help="Output directory "
                        "(default: out/compare/<comparison name>); preserves earlier reports.")


def _add_gate_parser(commands: argparse._SubParsersAction[LabArgumentParser]) -> None:
    parser = commands.add_parser(
        "gate", help="Evaluate a Policy Trace Lab rollout gate.",
        description="Policy Trace Lab evaluates recorded paired runs against gate rules.",
        epilog="Exit codes: 0 pass, 1 regression, 2 invalid/incomplete/blocked, 3 usage error "
               "or internal error.",
    )
    parser.add_argument("compare_json", type=Path, help="Recorded compare.json file.")
    parser.add_argument("--gate", type=Path, required=True, help="Gate configuration JSON file.")
    parser.add_argument("--out", type=Path,
                        help="Report directory (default: out/gate/<gate name>); "
                             "preserves earlier reports.")


def _add_report_parser(commands: argparse._SubParsersAction[LabArgumentParser]) -> None:
    parser = commands.add_parser(
        "report", help="Write an offline Policy Trace Lab replay report.",
        description="Policy Trace Lab renders recorded comparison and optional gate evidence "
                    "with paired trace replay into one offline report.html file.",
        epilog="Exit codes: 0 written, 1 findings/regression (reserved), "
               "2 invalid/unreadable/incomplete input or overwrite refusal, "
               "3 usage or internal error.",
    )
    parser.add_argument("compare_json", type=Path, help="Recorded compare.json file.")
    parser.add_argument("--gate", type=Path, help="Recorded gate.json result file.")
    parser.add_argument("--out", type=Path,
                        help="Output directory (default: out/report/<comparison name>).")


def _add_demo_parser(commands: argparse._SubParsersAction[LabArgumentParser]) -> None:
    parser = commands.add_parser(
        "demo", help="Run the offline Policy Trace Lab synthetic fixture demo.",
        description="Policy Trace Lab synthetic fixture demo needs only the CPU and the "
                    "default install. Synthetic fixture data. Not a benchmark.",
        epilog="Exit codes: 0 pass, 1 regression, 2 invalid/incomplete, 3 usage error.",
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory; preserves files.")
    parser.add_argument("--task", default="pusht", help="Task adapter name (default: pusht).")


def _report_file(compare_path: Path, gate_path: Path | None, out: Path | None) -> Path:
    comparison = report.load_result(compare_path)
    result = report.load_result(gate_path, gate_result=True) if gate_path is not None else None
    page = report.render_report(comparison, result, base_dir=compare_path.parent)
    directory = out if out is not None else Path("out/report") / comparison["comparison"]["name"]
    output = directory / "report.html"
    adapters.write_report_file(output, page)
    return output


def _gate_document(compare_path: Path, gate_path: Path, out: Path | None) -> dict[str, Any]:
    config = schema.load_gate_config(gate_path)
    comparison = pairing.load_comparison_result(compare_path)
    document = gate.evaluate_gate(comparison, config, base_dir=compare_path.parent)
    directory = out if out is not None else Path("out/gate") / config.name
    adapters.write_report_file(directory / "gate.json", schema.canonical_json(document))
    return document


def _compare_document(
    baseline: Path, candidate: Path, config_path: Path, out: Path | None,
) -> tuple[dict[str, Any], Path]:
    config = schema.load_comparison_config(config_path)
    directory = out if out is not None else Path("out/compare") / config.name
    document = pairing.build_comparison(baseline, candidate, config, out_dir=directory)
    output = directory / "compare.json"
    adapters.write_report_file(output, schema.canonical_json(document))
    return document, output


def _run_report(args: argparse.Namespace) -> int:
    output = _report_file(args.compare_json, args.gate, args.out)
    sys.stdout.write(f"report.html: {output}\n")
    return _EXIT_OK


def _run_gate(args: argparse.Namespace) -> int:
    document = _gate_document(args.compare_json, args.gate, args.out)
    out = args.out if args.out is not None else Path("out/gate") / document["gate"]["name"]
    sys.stdout.write(gate.render_verdict(document) + f"\ngate.json: {out / 'gate.json'}\n")
    return gate.gate_exit_code(document)


def _run_compare(args: argparse.Namespace) -> int:
    document, output = _compare_document(args.baseline_run, args.candidate_run,
                                         args.comparison, args.out)
    sys.stdout.write(pairing.render_summary(document) + f"\ncompare.json: {output}\n")
    for reason in document["invalid_reasons"]:
        sys.stderr.write(reason + "\n")
    return pairing.compare_exit_code(document)


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


def _doctor_document(path: Path, adapter: TaskAdapter, out: Path) -> dict[str, Any]:
    if not path.exists():
        raise schema.TraceReadError(f"Input does not exist: {path}")
    episodes = (adapters.read_traces(path) if adapters.is_trace_dir(path)
                else adapter.read_demonstrations(path))
    document = doctor.check_episodes(episodes, input_path=str(path),
                                      input_checksum=adapters.path_checksum(path))
    adapters.write_report_file(out / "doctor.json", schema.canonical_json(document))
    return document


def _run_doctor(args: argparse.Namespace, adapter: TaskAdapter) -> int:
    document = _doctor_document(args.input, adapter, args.out)
    sys.stdout.write(doctor.render_summary(document)
                     + f"\ndoctor.json: {args.out / 'doctor.json'}\n")
    return doctor.exit_code(document)


def _demo_fixtures(adapter: TaskAdapter) -> FixtureSet:
    fixtures = adapter.fixtures()
    scenario = fixtures.scenario
    required = [scenario / name for name in ("comparison.json", "gate.json")]
    directories = (fixtures.demo_clean, fixtures.demo_corrupt,
                   scenario / "baseline", scenario / "candidate")
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(_FIXTURES_MISSING)
        episodes = [path for path in directory.iterdir() if path.is_dir()]
        if not episodes:
            raise FileNotFoundError(_FIXTURES_MISSING)
        required.extend(path / name for path in episodes
                        for name in ("manifest.json", "steps.jsonl"))
    for variant in ("baseline", "candidate"):
        directory = scenario / variant
        required.extend(directory / name for name in ("run_manifest.json", "resolved_config.json"))
        try:
            manifest = schema.read_run_manifest(directory)
        except schema.TraceReadError as error:
            raise FileNotFoundError(_FIXTURES_MISSING) from error
        required.extend(directory / cell["episode_id"] / name for cell in manifest["episodes"]
                        for name in ("manifest.json", "steps.jsonl"))
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(_FIXTURES_MISSING)
    return fixtures


def _run_demo(args: argparse.Namespace, adapter: TaskAdapter) -> int:
    fixtures = _demo_fixtures(adapter)
    out: Path = args.out
    scenario = out / "fixtures/scenario"
    for source, target in ((fixtures.scenario, scenario),
                           (fixtures.demo_clean, out / "fixtures/demos/clean"),
                           (fixtures.demo_corrupt, out / "fixtures/demos/corrupt")):
        adapters.copy_tree_once(source, target)
    summaries = []
    for name in ("clean", "corrupt"):
        document = _doctor_document(out / "fixtures/demos" / name, adapter, out / "doctor" / name)
        summaries.append(f"== Doctor: {name} fixture (synthetic) ==\n"
                         + doctor.render_summary(document))
    _, comparison = _compare_document(scenario / "baseline", scenario / "candidate",
                                       scenario / "comparison.json", out / "compare")
    result = _gate_document(comparison, scenario / "gate.json", out / "gate")
    page = _report_file(comparison, out / "gate/gate.json", out / "report")
    lines = ["== Gate (synthetic fixture, not a benchmark) ==", gate.render_verdict(result),
             f"report.html: {page}"]
    if result["regressions"]:
        regression = result["regressions"][0]
        key = regression["key"]
        lines.append(f"open: {page}#pair={key['condition']}:{key['seed']}"
                     f"&step={regression['replay_step']}")
    code = gate.gate_exit_code(result)
    sys.stdout.write("\n\n".join([*summaries, "\n".join(lines)]) + f"\ndemo exit code: {code}\n")
    return code


def run_lab(args: argparse.Namespace) -> int:
    """Dispatch a lab command and translate input/usage/internal failures to exit codes."""
    if args.lab_command not in ("policies", "doctor", "run", "compare", "gate", "report", "demo"):
        sys.stderr.write(f"Unknown Policy Trace Lab command: {args.lab_command}\n")
        return _EXIT_USAGE_OR_INTERNAL
    try:
        handlers = {"report": _run_report, "gate": _run_gate,
                    "compare": _run_compare, "run": _run_export}
        if args.lab_command in handlers:
            return handlers[args.lab_command](args)
        try:
            adapter = get_task(args.task)
        except KeyError as error:
            sys.stderr.write(f"{error}\n")
            return _EXIT_USAGE_OR_INTERNAL
        if args.lab_command == "demo":
            return _run_demo(args, adapter)
        if args.lab_command == "doctor":
            return _run_doctor(args, adapter)
        return _run_policies(adapter)
    except (OSError, schema.ConfigError, schema.TraceReadError, exporter.SetupError) as error:
        sys.stderr.write(f"Cannot read or write Policy Trace Lab input/output: {error}\n")
        return _EXIT_INVALID_INPUT
    except Exception as error:
        sys.stderr.write(f"Policy Trace Lab failed: {error}\n")
        return _EXIT_USAGE_OR_INTERNAL
