"""Regenerate JSON Schemas with python -m guided_streaming_policy.lab.schema."""

from .jsonschemas import main

__all__: list[str] = []

if __name__ == "__main__":
    raise SystemExit(main())
