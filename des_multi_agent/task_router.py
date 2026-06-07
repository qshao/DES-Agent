from __future__ import annotations

import json
import os

import yaml
from dataclasses import fields

from .config import PROJECT_ROOT
from .llm.config import LLMConfig
from .llm.factory import build_llm_provider
from .llm.parser import extract_json_object
from .task_router_schema import RouterJob, RouterResponse

DEFAULT_ROUTER_LLM_CONFIG = PROJECT_ROOT / "llm.example.yaml"


def _normalize_questions(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def build_default_router_provider():
    raw = yaml.safe_load(DEFAULT_ROUTER_LLM_CONFIG.read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict) and "llm" in raw and isinstance(raw["llm"], dict):
        raw = raw["llm"]
    if not isinstance(raw, dict):
        raise ValueError(f"Router LLM config file {DEFAULT_ROUTER_LLM_CONFIG} must contain a mapping")
    cfg = LLMConfig.from_mapping(raw)
    cfg.validate()
    provider = build_llm_provider(cfg)
    if provider is None:
        raise ValueError("task-router requires an enabled LLM config")
    return provider


def parse_router_response(payload: str) -> RouterResponse:
    try:
        raw = json.loads(extract_json_object(payload))
    except Exception as exc:
        raise ValueError(f"Failed to parse router response: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("router response must be a JSON object")
    workflow = str(raw.get("workflow", "")).strip().lower()
    needs_clarification = bool(raw.get("needs_clarification", False))
    clarifying_questions = _normalize_questions(raw.get("clarifying_questions"))
    job_data = raw.get("job")
    if isinstance(job_data, dict):
        allowed_fields = {item.name for item in fields(RouterJob)}
        filtered_job_data = {key: value for key, value in job_data.items() if key in allowed_fields}
        try:
            job = RouterJob(**filtered_job_data)
        except TypeError as exc:
            raise ValueError(f"router response job fields are invalid: {exc}") from exc
    else:
        job = None
    if needs_clarification:
        if not clarifying_questions:
            raise ValueError("router response requested clarification but provided no questions")
        if workflow not in {"des", "metal-binding"}:
            workflow = "clarify"
        response = RouterResponse(
            workflow=workflow,
            needs_clarification=True,
            clarifying_questions=clarifying_questions,
            job=job,
        )
        response.validate()
        return response
    if clarifying_questions:
        raise ValueError("router response included clarifying questions without requesting clarification")
    if workflow not in {"des", "metal-binding"}:
        raise ValueError(f"Unsupported workflow: {workflow}")
    if job is None:
        raise ValueError("router response must include a job when clarification is not needed")
    response = RouterResponse(
        workflow=workflow,
        needs_clarification=False,
        clarifying_questions=[],
        job=job,
    )
    response.validate()
    return response


def route_task(request: str, provider=None) -> RouterResponse:
    provider = provider or build_default_router_provider()
    response_text = provider.route_request(request)
    return parse_router_response(response_text)
