from functools import lru_cache
from pathlib import Path
import os


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] in {'"', "'"} and value[-1:] == value[0]:
            value = value[1:-1]

        os.environ.setdefault(key, value)


class Settings:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        self.base_dir = base_dir
        _load_dotenv(base_dir / ".env")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        self.mistral_model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        self.mistral_base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/")
        self.upload_dir = base_dir / os.getenv("UPLOAD_DIR", "runtime/uploads")
        self.output_dir = base_dir / os.getenv("OUTPUT_DIR", "runtime/output")
        self.jobs_dir = base_dir / os.getenv("JOBS_DIR", "runtime/jobs")
        self.tracker_path = base_dir / os.getenv("TRACKER_PATH", "runtime/tracker.json")
        self.max_pages_per_chunk = int(os.getenv("MAX_PAGES_PER_CHUNK", "8"))

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
