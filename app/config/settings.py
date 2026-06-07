import os
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv, set_key


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"

REQUIRED_ENV_VARS = (
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODELS = (
    DEFAULT_GROQ_MODEL,
    "llama-3.1-8b-instant",
    "llama-3.3-70b-specdec",
)

OPENROUTER_EMBEDDING_MODEL = "google/gemini-embedding-001"
OPENROUTER_EMBEDDING_DIMENSIONS = 1536
OPENROUTER_CHAT_MODEL = "google/gemini-2.5-flash"


load_dotenv(ENV_FILE)


def get_setting(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_runtime_settings() -> dict[str, str]:
    return {name: get_setting(name) for name in REQUIRED_ENV_VARS}


def missing_required_settings(settings: Mapping[str, str]) -> list[str]:
    return [
        name
        for name in REQUIRED_ENV_VARS
        if not settings.get(name, "").strip()
    ]


def apply_runtime_settings(settings: Mapping[str, str]) -> None:
    for name in REQUIRED_ENV_VARS:
        value = settings.get(name, "").strip()
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)


def save_settings_to_env(settings: Mapping[str, str]) -> None:
    ENV_FILE.touch(exist_ok=True)

    for name in REQUIRED_ENV_VARS:
        set_key(
            str(ENV_FILE),
            name,
            settings.get(name, "").strip(),
        )
