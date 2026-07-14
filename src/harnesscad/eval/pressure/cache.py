"""Content-addressed disk cache for model outputs.

A pressure run is a few hundred ollama calls; without a cache, re-running the
report means re-running the models, and "reproducible" quietly becomes "roughly
the same". The key is the full determinant of the completion --
(model, seed, temperature, attempt index, and the exact message list) -- hashed
to a sha256. Same inputs, same file, same bytes: a second run is free and
byte-identical, and a run that was interrupted resumes where it stopped.

The cache is a plain directory of JSON files. stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional


def cache_key(model: str, seed: int, temperature: float, attempt: int,
              messages: List[Dict[str, Any]]) -> str:
    """The sha256 of everything that determines the completion."""
    blob = json.dumps(
        {
            "model": model,
            "seed": seed,
            "temperature": temperature,
            "attempt": attempt,
            "messages": messages,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CompletionCache:
    """A directory of ``<sha256>.json`` completions.

    ``get`` returns None on a miss (and on a corrupt entry, which is treated as a
    miss rather than an error -- a half-written file from a killed run must not
    poison a resume).
    """

    def __init__(self, root: str) -> None:
        self.root = str(root)
        self.hits = 0
        self.misses = 0
        os.makedirs(self.root, exist_ok=True)

    def path(self, key: str) -> str:
        return os.path.join(self.root, key + ".json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self.path(key)
        if not os.path.exists(p):
            self.misses += 1
            return None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, key: str, entry: Dict[str, Any]) -> None:
        # Write-then-rename: an interrupted run leaves no half-parsed entry.
        p = self.path(key)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, sort_keys=True, indent=2)
        os.replace(tmp, p)

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
