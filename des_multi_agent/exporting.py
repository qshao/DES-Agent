from __future__ import annotations

from collections.abc import Mapping
from csv import DictWriter
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
import json


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


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(content)
    try:
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _build_manifest(output_dir: Path, run_payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "workflow": run_payload.get("workflow"),
        "component_a": run_payload.get("component_a"),
        "n": run_payload.get("n"),
        "exported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "json_filename": "run.json",
        "csv_filename": "run.csv",
        "manifest_filename": "run.manifest.json",
    }


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


def export_des_run_bundle(output_dir: str | Path, run_payload: Mapping[str, object]) -> dict[str, Path]:
    output_path = Path(output_dir)
    payload = _require_mapping(run_payload, "run_payload")
    rows = _build_csv_rows(payload)
    manifest = _build_manifest(output_path, payload)

    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "run.json"
    csv_path = output_path / "run.csv"
    manifest_path = output_path / "run.manifest.json"

    json_text = json.dumps(_json_safe(payload), indent=2, sort_keys=True)
    csv_buffer = StringIO()
    writer = DictWriter(csv_buffer, fieldnames=list(CSV_COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    manifest_text = json.dumps(_json_safe(manifest), indent=2, sort_keys=True)

    _write_text_atomic(json_path, json_text)
    _write_text_atomic(csv_path, csv_buffer.getvalue())
    _write_text_atomic(manifest_path, manifest_text)

    return {"json": json_path, "csv": csv_path, "manifest": manifest_path}
