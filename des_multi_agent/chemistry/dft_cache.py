"""SQLite-backed cache for DFT results, keyed by (species_smiles, dft_method).

Entry point: cached_compute_dft_properties(smiles, pH=7.0, dft_method=..., cache_path=None).
Never raises — any cache-layer failure falls back to an uncached compute_dft_properties call.
Only success=True results are cached.
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
import time
from pathlib import Path

from .dft_validator import DFTResult, DEFAULT_DFT_METHOD, compute_dft_properties
from .protonation import dominant_species

DEFAULT_CACHE_PATH: Path = (
    Path(__file__).resolve().parents[2] / "artifacts" / "dft_cache" / "dft_results.sqlite3"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dft_cache (
    species_smiles TEXT NOT NULL,
    dft_method     TEXT NOT NULL,
    result_json    TEXT NOT NULL,
    computed_at    REAL NOT NULL,
    PRIMARY KEY (species_smiles, dft_method)
)
"""


def _resolve_cache_path(cache_path: str | Path | None) -> Path:
    return Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    try:
        conn.execute(_SCHEMA)
    except Exception:
        conn.close()
        raise
    return conn


def _load_cached(conn: sqlite3.Connection, species_smiles: str, dft_method: str) -> DFTResult | None:
    row = conn.execute(
        "SELECT result_json FROM dft_cache WHERE species_smiles = ? AND dft_method = ?",
        (species_smiles, dft_method),
    ).fetchone()
    if row is None:
        return None
    return DFTResult(**json.loads(row[0]))


def _store(conn: sqlite3.Connection, species_smiles: str, dft_method: str,
           result: DFTResult, computed_at: float) -> None:
    payload = json.dumps(dataclasses.asdict(result))
    conn.execute(
        "INSERT OR REPLACE INTO dft_cache "
        "(species_smiles, dft_method, result_json, computed_at) VALUES (?, ?, ?, ?)",
        (species_smiles, dft_method, payload, computed_at),
    )
    conn.commit()


def cached_compute_dft_properties(
    smiles: str,
    pH: float = 7.0,
    dft_method: str = DEFAULT_DFT_METHOD,
    cache_path: str | Path | None = None,
) -> DFTResult:
    """Cache-aware wrapper around compute_dft_properties. Never raises."""
    try:
        species_smiles = dominant_species(smiles, pH).species_smiles
    except Exception:
        species_smiles = smiles

    resolved_path = _resolve_cache_path(cache_path)

    conn = None
    try:
        conn = _connect(resolved_path)
        cached = _load_cached(conn, species_smiles, dft_method)
        if cached is not None:
            cached.from_cache = True
            cached.smiles = smiles
            cached.ph = pH
            return cached
    except Exception:
        pass
    finally:
        if conn is not None:
            conn.close()

    result = compute_dft_properties(smiles, pH=pH)

    if result.success:
        conn2 = None
        try:
            conn2 = _connect(resolved_path)
            _store(conn2, species_smiles, dft_method, result, time.time())
        except Exception:
            pass
        finally:
            if conn2 is not None:
                conn2.close()

    return result
