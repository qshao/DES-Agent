from __future__ import annotations

import json
from urllib import request as urllib_request


def post_json_chat(
    url: str,
    payload: dict,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
) -> str:
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=body, headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib_request.urlopen(req, timeout=timeout_seconds) as resp:
        return resp.read().decode("utf-8")
