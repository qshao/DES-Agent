from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .cache import LLMCache
from .client import post_json_chat


@dataclass(frozen=True)
class RequestTransport:
    request_fn: object = post_json_chat
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    cache_dir: str | None = None
    cache_ttl_seconds: float = 3600.0

    def post_json(
        self,
        url: str,
        payload: dict,
        *,
        api_key: str | None = None,
        include_api_key_in_header: bool = True,
    ) -> str:
        header_key = api_key if include_api_key_in_header else None

        def _call(u, p, **kw):
            attempts = max(1, self.max_retries)
            last_exc: Exception | None = None
            for attempt in range(attempts):
                try:
                    return self.request_fn(u, p, api_key=header_key,
                                           timeout_seconds=self.timeout_seconds)
                except OSError as exc:
                    last_exc = exc
                    if attempt < attempts - 1:
                        delay = (self.retry_base_delay * (2 ** attempt)
                                 + random.uniform(0.0, 0.5))
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return LLMCache(cache_dir=self.cache_dir,
                        ttl_seconds=self.cache_ttl_seconds).get_or_call(url, payload, _call)
