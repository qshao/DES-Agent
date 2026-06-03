from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestProfile:
    name: str
    path_template: str
    payload_style: str
    api_key_in_header: bool = True
    api_key_in_query: bool = False
