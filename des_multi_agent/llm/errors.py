from __future__ import annotations

import json


class LLMSchemaError(ValueError):
    """Raised when an LLM response parses as valid JSON but fails structural validation."""


def payload_excerpt(raw: str, limit: int = 160) -> str:
    text = raw.replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def response_error(provider_name: str, expectation: str, raw: str) -> ValueError:
    return ValueError(f"{provider_name} response {expectation}. Payload excerpt: {payload_excerpt(raw)}")


def load_json_or_raise(provider_name: str, raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise response_error(provider_name, "must be valid JSON", raw) from exc
