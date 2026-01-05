from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

OUTPUTS_SALESOVERVIEW = BASE_DIR / "outputs_salesoverview"
OUTPUTS_TITLES_DISTRIBUTORS = BASE_DIR / "outputs_titlesdistributors"
OUTPUTS_LOCATION_QUESTIONS = BASE_DIR / "outputs_locationquestions"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_database_path(preferred: Optional[Path] = None) -> Path:
    candidates = []
    if preferred is not None:
        candidates.append(preferred)

    candidates.extend(sorted(DATA_DIR.glob("*.sqlite")))
    candidates.extend(sorted(DATA_DIR.glob("*.db")))
    candidates.extend(sorted(BASE_DIR.glob("*.sqlite")))
    candidates.extend(sorted(BASE_DIR.glob("*.db")))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("No .db/.sqlite file found in project root or data/")
