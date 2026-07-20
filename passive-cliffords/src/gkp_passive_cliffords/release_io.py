"""Transparent reads for development JSON and compressed release artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def load_json_artifact(path: str | Path):
    """Load ``path`` or, when absent, the corresponding ``path.json.gz``."""

    plain = Path(path)
    if plain.exists():
        return json.loads(plain.read_text())
    compressed = Path(str(plain) + ".gz")
    if compressed.exists():
        with gzip.open(compressed, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    raise FileNotFoundError(f"neither {plain} nor {compressed} exists")
