"""B4 — LLM call caching.

Disk cache keyed on SHA-256 of (url, json-serialised payload).
A TTL of 0 disables caching. cache_dir=None disables caching entirely.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class LLMCache:
    def __init__(self, cache_dir: str | Path | None, ttl_seconds: float = 3600.0) -> None:
        self._dir = Path(cache_dir) if cache_dir is not None else None
        self._ttl = ttl_seconds

    def _key(self, url: str, payload: dict) -> str:
        raw = json.dumps({"url": url, "payload": payload}, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        assert self._dir is not None
        return self._dir / f"{key}.json"

    def _is_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self._ttl <= 0:
            return False
        age = time.time() - path.stat().st_mtime
        return age < self._ttl

    def get_or_call(self, url: str, payload: dict, backend, **backend_kwargs) -> str:
        if self._dir is None:
            return backend(url, payload, **backend_kwargs)

        key = self._key(url, payload)
        path = self._cache_path(key)

        if self._is_valid(path):
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        response = backend(url, payload, **backend_kwargs)

        self._dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"response": response}), encoding="utf-8")
        return response
