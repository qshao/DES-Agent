from __future__ import annotations

from .base import BaseLLMProvider
from .errors import load_json_or_raise, response_error
from .specs import RequestProfile


class NemotronProvider(BaseLLMProvider):
    request_profile = RequestProfile(
        name="Nemotron",
        path_template="/api/chat",
        payload_style="ollama",
        api_key_in_header=True,
        api_key_in_query=False,
    )

    def extract_text(self, raw: str) -> str:
        data = load_json_or_raise("Nemotron", raw)
        if not isinstance(data, dict):
            raise response_error("Nemotron", "must return a JSON object with message.content", raw)
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        raise response_error("Nemotron", "is missing message.content", raw)
