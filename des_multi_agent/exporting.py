from __future__ import annotations

from collections.abc import Mapping
from csv import DictWriter
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from .memory_schema import RunCandidateSummary


CSV_COLUMNS = (
    "smiles_b",
    "is_des",
    "min_tm_k",
    "rank",
    "source",
    "source_id",
    "trust_score",
    "uncertainty_flag",
)

_CANDIDATE_SUMMARY_FIELDS = frozenset(f.name for f in dataclass_fields(RunCandidateSummary))
_CSV_CANDIDATE_FIELDS = frozenset(CSV_COLUMNS) - {"is_des"}
if _CANDIDATE_SUMMARY_FIELDS != _CSV_CANDIDATE_FIELDS:
    _missing = _CANDIDATE_SUMMARY_FIELDS - _CSV_CANDIDATE_FIELDS
    _extra = _CSV_CANDIDATE_FIELDS - _CANDIDATE_SUMMARY_FIELDS
    raise RuntimeError(
        f"CSV_COLUMNS is out of sync with RunCandidateSummary fields — "
        f"missing={_missing}, extra={_extra}"
    )


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_csv_field(result: Mapping[str, object], field: str, index: int) -> object:
    if field not in result:
        raise KeyError(f"result[{index}] is missing required CSV field {field!r}")
    return result[field]


def _coerce_csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return None
    return value


def _build_manifest(output_dir: Path, run_payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "workflow": run_payload.get("workflow"),
        "component_a": run_payload.get("component_a"),
        "n": run_payload.get("n"),
        "exported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "report_filename": "report.txt",
        "json_filename": "run.json",
        "csv_filename": "run.csv",
        "manifest_filename": "run.manifest.json",
    }


def _replace_export_file(staged_path: Path, final_path: Path) -> None:
    staged_path.replace(final_path)


def _build_csv_rows(run_payload: Mapping[str, object]) -> list[dict[str, str]]:
    results = run_payload.get("results", [])
    if not isinstance(results, list):
        raise TypeError("run_payload['results'] must be a list")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(results):
        result = _require_mapping(item, f"run_payload['results'][{index}]")
        row = {}
        for field in CSV_COLUMNS:
            row[field] = _coerce_csv_value(_require_csv_field(result, field, index))
        rows.append(row)
    return rows


def export_des_run_bundle(output_dir: str | Path, run_payload: Mapping[str, object], report_text: str) -> dict[str, Path]:
    output_path = Path(output_dir)
    payload = _require_mapping(run_payload, "run_payload")
    rows = _build_csv_rows(payload)
    manifest = _build_manifest(output_path, payload)

    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "report.txt"
    json_path = output_path / "run.json"
    csv_path = output_path / "run.csv"
    manifest_path = output_path / "run.manifest.json"

    json_text = json.dumps(_json_safe(payload), indent=2, sort_keys=True)
    csv_buffer = StringIO()
    writer = DictWriter(csv_buffer, fieldnames=list(CSV_COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    manifest_text = json.dumps(_json_safe(manifest), indent=2, sort_keys=True)

    staged_content = {
        "report.txt": report_text,
        "run.json": json_text,
        "run.csv": csv_buffer.getvalue(),
        "run.manifest.json": manifest_text,
    }
    final_paths = {
        "report.txt": report_path,
        "run.json": json_path,
        "run.csv": csv_path,
        "run.manifest.json": manifest_path,
    }

    with TemporaryDirectory(prefix=".export-", dir=output_path) as stage_dir_name:
        stage_dir = Path(stage_dir_name)
        staged_paths: dict[str, Path] = {}
        for filename, content in staged_content.items():
            staged_path = stage_dir / filename
            staged_path.write_text(content, encoding="utf-8")
            staged_paths[filename] = staged_path
        for filename in ("report.txt", "run.json", "run.csv", "run.manifest.json"):
            _replace_export_file(staged_paths[filename], final_paths[filename])

    return {"report": report_path, "json": json_path, "csv": csv_path, "manifest": manifest_path}
