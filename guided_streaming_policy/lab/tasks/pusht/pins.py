"""Compute simulator, checkpoint-byte and resolved-default configuration pins.

Regenerate from the repository root with this one-line command:
.venv/bin/python -c "from pathlib import Path; from guided_streaming_policy.lab.schema import canonical_json; from guided_streaming_policy.lab.tasks.pusht.pins import pin_file_document; Path('guided_streaming_policy/lab/tasks/pusht/pins.json').write_text(canonical_json(pin_file_document(Path.cwd())), encoding='utf-8')"

The task section contains simulator {name, version} and policies keyed by name;
each policy holds checkpoints {repo-relative path: SHA-256 or null} and config_sha256.
"""  # noqa: E501

import importlib.metadata
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import adapters, schema
from guided_streaming_policy.lab.tasks.pusht.policies import POLICIES

__all__ = ["compute_pins", "pin_file_document"]


def compute_pins(root: Path) -> dict[str, Any]:
    """Recompute task pins from disk, keeping absent checkpoints explicitly null."""
    try:
        version = importlib.metadata.version("pymunk")
    except importlib.metadata.PackageNotFoundError:
        version = None
    checkpoint_hashes = {
        path: schema.sha256_file(root / path) if (root / path).is_file() else None
        for path in sorted({path for spec in POLICIES for path in spec.checkpoints})
    }
    policies = {
        spec.name: {
            "checkpoints": {path: checkpoint_hashes[path] for path in spec.checkpoints},
            "config_sha256": adapters.config_sha256(adapters.resolve_policy_config(spec, {})),
        }
        for spec in POLICIES
    }
    return {"simulator": {"name": "pymunk", "version": version}, "policies": policies}


def pin_file_document(root: Path) -> dict[str, Any]:
    """Wrap the task pins in a versioned document with a tasks map."""
    return {"schema_version": schema.SCHEMA_VERSION, "tasks": {"pusht": compute_pins(root)}}
