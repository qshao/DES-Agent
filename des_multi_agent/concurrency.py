from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CallResult(Generic[T]):
    value: T | None
    error: Exception | None


def run_concurrent(items: list[Any], call: Callable[[Any], T], max_workers: int = 8) -> list[CallResult[T]]:
    """Run call(item) for every item concurrently; never raises.

    Results are returned in the same order as items (ThreadPoolExecutor.map
    preserves input order), so callers can zip(items, run_concurrent(...))
    to line results up with what produced them. Each call is wrapped so a
    single item's exception is captured as CallResult.error instead of
    aborting the others.
    """
    if not items:
        return []

    def _safe_call(item: Any) -> CallResult[T]:
        try:
            return CallResult(value=call(item), error=None)
        except Exception as exc:  # noqa: BLE001 - intentionally broad, mirrors existing per-item try/except
            return CallResult(value=None, error=exc)

    workers = min(len(items), max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_safe_call, items))
