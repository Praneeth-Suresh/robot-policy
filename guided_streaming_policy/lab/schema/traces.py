"""Trace constants and canonical IO, independent of task and third-party code."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "KIND_DEMONSTRATION",
    "KIND_ROLLOUT",
    "SEVERITIES",
    "MANIFEST_FILENAME",
    "STEPS_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "FIELDS_PRESENT_VALUES",
    "required_manifest_fields",
    "Episode",
    "TraceReadError",
    "canonical_json",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "write_episode",
    "read_episode",
    "list_episode_dirs",
    "write_run_manifest",
    "read_run_manifest",
    "make_provenance",
]

SCHEMA_VERSION = "0.1"
KIND_DEMONSTRATION = "demonstration"
KIND_ROLLOUT = "rollout"
SEVERITIES = ("error", "warning", "info")
MANIFEST_FILENAME = "manifest.json"
STEPS_FILENAME = "steps.jsonl"
RUN_MANIFEST_FILENAME = "run_manifest.json"
FIELDS_PRESENT_VALUES = (
    "obs",
    "action_executed",
    "action_proposed",
    "reward",
    "info",
    "reset_state_fingerprint",
    "geometry",
)
_COMMON_FIELDS = (
    "schema_version",
    "kind",
    "episode_id",
    "source_episode_id",
    "task",
    "split",
    "split_source",
    "timebase",
    "observation_schema",
    "action_schema",
    "fields_present",
    "provenance",
)
_ROLLOUT_FIELDS = (
    "condition",
    "seed",
    "reset_state_fingerprint",
    "policy",
    "resolved_config",
    "simulator",
    "outcome",
)
_JSON_INDENT_SPACES = 2
_HASH_CHUNK_BYTES = 65536
_GIT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class Episode:
    """A manifest and its ordered steps; no telemetry is synthesized."""

    manifest: dict[str, Any]
    steps: tuple[dict[str, Any], ...]


class TraceReadError(Exception):
    """A trace is missing, unreadable, or contains invalid JSON objects."""


def make_provenance(
    command: str,
    created_at: str | None = None,
    *,
    root: Path | None = None,
) -> dict[str, str]:
    """Record tool version, Git commit, command and UTC creation time to seconds."""
    try:
        tool_version = importlib.metadata.version("guided-streaming-policy")
    except importlib.metadata.PackageNotFoundError:
        tool_version = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3] if root is None else root,
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
        git_commit = result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        git_commit = "unknown"
    return {
        "tool_version": tool_version,
        "git_commit": git_commit,
        "command": command,
        "created_at": created_at
        if created_at is not None
        else datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def required_manifest_fields(kind: str) -> tuple[str, ...]:
    """Return common plus kind-specific mandatory manifest keys."""
    if kind == KIND_DEMONSTRATION:
        return _COMMON_FIELDS + ("dataset",)
    if kind == KIND_ROLLOUT:
        return _COMMON_FIELDS + _ROLLOUT_FIELDS
    raise ValueError(f"Unknown episode kind: {kind!r}")


def canonical_json(obj: Any) -> str:
    """Serialize stable, readable UTF-8 JSON text, rejecting nonfinite numbers."""
    return (
        json.dumps(
            obj, sort_keys=True, indent=_JSON_INDENT_SPACES, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )


def _compact_json(obj: Any, *, allow_nan: bool = False) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=allow_nan
    )


def sha256_bytes(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash file bytes without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash compact sorted-key UTF-8 JSON, without a trailing newline."""
    return sha256_bytes(_compact_json(obj).encode("utf-8"))


def write_episode(episode: Episode, directory: Path) -> Path:
    """Create a new episode directory; preserve corrupt step numbers for Doctor."""
    episode_id = episode.manifest["episode_id"]
    if (
        not isinstance(episode_id, str)
        or not episode_id
        or episode_id in (".", "..")
        or Path(episode_id).name != episode_id
    ):
        raise ValueError("episode_id must be a single directory name")
    manifest_text = canonical_json(episode.manifest)
    target = directory / episode_id
    target.mkdir(parents=True, exist_ok=False)
    (target / MANIFEST_FILENAME).write_text(manifest_text, encoding="utf-8")
    with (target / STEPS_FILENAME).open("w", encoding="utf-8") as stream:
        for step in episode.steps:
            stream.write(_compact_json(step, allow_nan=True) + "\n")
    return target


def _json_object(text: str) -> dict[str, Any]:
    value: Any = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def read_episode(path: Path) -> Episode:
    """Read JSON objects without schema validation; retain NaN for diagnostics."""
    try:
        manifest = _json_object((path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        with (path / STEPS_FILENAME).open(encoding="utf-8") as stream:
            steps = tuple(_json_object(line) for line in stream)
        return Episode(manifest, steps)
    except (OSError, ValueError) as error:
        raise TraceReadError(f"Cannot read episode {path}: {error}") from error


def list_episode_dirs(run_dir: Path) -> list[Path]:
    """List episode directories containing a manifest, sorted by directory name."""
    return sorted(
        (
            path
            for path in run_dir.iterdir()
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        ),
        key=lambda p: p.name,
    )


def write_run_manifest(manifest: dict[str, Any], run_dir: Path) -> Path:
    """Write the run's index as canonical JSON."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_MANIFEST_FILENAME
    path.write_text(canonical_json(manifest), encoding="utf-8")
    return path


def read_run_manifest(run_dir: Path) -> dict[str, Any]:
    """Read a run index, translating unreadable or invalid objects to TraceReadError."""
    path = run_dir / RUN_MANIFEST_FILENAME
    try:
        return _json_object(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TraceReadError(f"Cannot read run manifest {path}: {error}") from error
