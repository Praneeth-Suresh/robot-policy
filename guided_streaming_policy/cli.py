import argparse

from guided_streaming_policy.doctor import render_environment_report
from guided_streaming_policy.pusht_legacy import TASKS, run_task, task_names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gsp",
        description="Repo helper CLI for the Guided Streaming Policy project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-env", help="Report workspace and dependency status.")
    subparsers.add_parser("list-tasks", help="List available Push-T tasks.")

    run_parser = subparsers.add_parser("run", help="Run a Push-T task.")
    run_parser.add_argument("task", choices=task_names(), help="Task name returned by `gsp list-tasks`.")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it.",
    )

    return parser


def _handle_check_env() -> int:
    print(render_environment_report())
    return 0


def _handle_list_tasks() -> int:
    for task in TASKS:
        print(f"{task.name:18s} {task.module:40s} {task.description}")
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    return run_task(args.task, dry_run=args.dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check-env":
        return _handle_check_env()
    if args.command == "list-tasks":
        return _handle_list_tasks()
    if args.command == "run":
        return _handle_run(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
