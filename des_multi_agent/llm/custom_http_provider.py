from __future__ import annotations

from .base import BaseLLMProvider
from .errors import load_json_or_raise, response_error
from .specs import RequestProfile


class CustomHTTPProvider(BaseLLMProvider):
    request_profile = RequestProfile(
        name="Custom HTTP",
        path_template="",
        payload_style="openai",
        api_key_in_header=True,
        api_key_in_query=False,
    )

    def extract_text(self, raw: str) -> str:
        data = load_json_or_raise("Custom HTTP", raw)
        if not isinstance(data, dict):
            raise response_error("Custom HTTP", "must return a JSON object with choices[0].message.content", raw)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise response_error("Custom HTTP", "is missing choices[0].message.content", raw)
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise response_error("Custom HTTP", "is missing choices[0].message.content", raw)
        content = message.get("content")
        if not isinstance(content, str):
            raise response_error("Custom HTTP", "is missing choices[0].message.content", raw)
        return content
