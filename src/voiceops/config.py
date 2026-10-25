"""Central configuration. Everything is overridable from the environment / .env."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _find_dataset_dir() -> Path:
    """Locate the extracted capstone dataset (the folder that contains kb/ and data/)."""
    candidates = []
    if os.getenv("DATASET_DIR"):
        candidates.append(Path(os.environ["DATASET_DIR"]))
    candidates += [ROOT / "dataset", ROOT / "Capstone_Project_Dataset_FTTT_2026"]
    for base in candidates:
        if (base / "kb").is_dir():
            return base.resolve()
        # The ZIP may extract into a nested folder.
        if base.is_dir():
            for sub in sorted(base.iterdir()):
                if (sub / "kb").is_dir():
                    return sub.resolve()
    return candidates[0].resolve()


DATASET_DIR = _find_dataset_dir()
KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"
TOOLS_DIR = DATASET_DIR / "tools"

CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# The dataset's reference date ("today") per the brief.
TODAY = date.fromisoformat(os.getenv("TODAY", "2026-10-20"))

COMPANY = "ZentroMart"

# LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or (
    "sarvam" if SARVAM_API_KEY else "gemini" if GEMINI_API_KEY else "anthropic")
_DEFAULT_MODEL = {"sarvam": "sarvam-105b", "gemini": "gemini-3.5-flash",
                  "anthropic": "claude-opus-5-5"}[LLM_PROVIDER]
LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODEL
LLM_EFFORT = os.getenv("LLM_EFFORT", "low")  # thinking level; low keeps voice turns snappy
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)
USE_FALLBACKS = os.getenv("LLM_FALLBACKS", "1") == "1"

# Retrieval
# Multilingual so Hindi / Hinglish questions land near the English policy text.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", "8"))

# Voice (Gnani)
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en-IN")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en-IN")
TTS_VOICE = os.getenv("TTS_VOICE", "Kaveri")
TTS_MODEL = os.getenv("TTS_MODEL", "timbre-v2.5")
