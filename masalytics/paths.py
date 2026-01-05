"""Project path helpers and output directory management."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_SALESOVERVIEW_DIR = PROJECT_ROOT / "outputs_salesoverview"
OUTPUTS_TITLES_DISTRIBUTORS_DIR = PROJECT_ROOT / "outputs_titlesdistributors"
OUTPUTS_LOCATIONQUESTIONS_DIR = PROJECT_ROOT / "outputs_locationquestions"

OUTPUT_DIRS = {
    "sales_overview": OUTPUTS_SALESOVERVIEW_DIR,
    "titles_distributors": OUTPUTS_TITLES_DISTRIBUTORS_DIR,
    "location_questions": OUTPUTS_LOCATIONQUESTIONS_DIR,
}


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    for path in OUTPUT_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def find_database(explicit_path: str | Path | None = None) -> Path:
    """Locate the SQLite database file for the project."""
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Database path not found: {candidate}")

    env_path = os.getenv("MASALYTICS_DB")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"MASALYTICS_DB path not found: {candidate}")

    candidates = [
        DATA_DIR / "numero_data.sqlite",
        DATA_DIR / "numero_data.db",
    ]
    candidates += sorted(DATA_DIR.glob("*.sqlite"))
    candidates += sorted(DATA_DIR.glob("*.db"))
    candidates += sorted(PROJECT_ROOT.glob("*.sqlite"))
    candidates += sorted(PROJECT_ROOT.glob("*.db"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No database file found. Set MASALYTICS_DB or place a .db/.sqlite file in data/."
    )
