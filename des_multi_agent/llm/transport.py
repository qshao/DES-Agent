from __future__ import annotations

from dataclasses import dataclass

from .client import post_json_chat


@dataclass(frozen=True)
class RequestTransport:
    request_fn: object = post_json_chat
    timeout_seconds: float = 30.0

    def post_json(
        self,
        url: str,
        payload: dict,
        *,
        api_key: str | None = None,
        include_api_key_in_header: bool = True,
    ) -> str:
        header_key = api_key if include_api_key_in_header else None
        return self.request_fn(
            url,
            payload,
            api_key=header_key,
            timeout_seconds=self.timeout_seconds,
        )
