from __future__ import annotations

from .base import BaseLLMProvider
from .errors import load_json_or_raise, response_error
from .specs import RequestProfile


class GeminiProvider(BaseLLMProvider):
    request_profile = RequestProfile(
        name="Gemini",
        path_template="/v1beta/models/{model_name}:generateContent",
        payload_style="gemini",
        api_key_in_header=False,
        api_key_in_query=True,
    )

    def extract_text(self, raw: str) -> str:
        data = load_json_or_raise("Gemini", raw)
        if not isinstance(data, dict):
            raise response_error("Gemini", "must return a JSON object with candidates[0].content.parts[0].text", raw)
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise response_error("Gemini", "is missing candidates[0].content.parts[0].text", raw)
        content = candidates[0].get("content")
        if not isinstance(content, dict):
            raise response_error("Gemini", "is missing candidates[0].content.parts[0].text", raw)
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise response_error("Gemini", "is missing candidates[0].content.parts[0].text", raw)
        text = parts[0].get("text")
        if not isinstance(text, str):
            raise response_error("Gemini", "is missing candidates[0].content.parts[0].text", raw)
        return text
