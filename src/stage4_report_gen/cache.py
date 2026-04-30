"""SHA-256-keyed disk cache for Gemini responses.

One JSON file per (model, prompt) pair, stored under the cache directory
configured in stage4_config.yaml :: gemini.cache_dir. The hash is over
both model name and prompt so that bumping GEMINI_MODEL invalidates the
cache implicitly.

Cache hits and misses are tracked on the instance so report_builder can
report a per-run cache hit rate.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GeminiCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(model: str, prompt: str) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\n")
        h.update(prompt.encode("utf-8"))
        return h.hexdigest()

    def _path(self, model: str, prompt: str) -> Path:
        return self.cache_dir / f"{self._key(model, prompt)}.json"

    def get(self, model: str, prompt: str) -> Optional[str]:
        path = self._path(model, prompt)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            response = payload["response"]
            self.hits += 1
            return response
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Corrupted cache entry %s — discarding: %s", path.name, exc)
            try:
                path.unlink()
            except OSError:
                pass
            self.misses += 1
            return None

    def put(self, model: str, prompt: str, response: str) -> None:
        path = self._path(model, prompt)
        payload = {"model": model, "response": response}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
