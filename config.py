import os
from pathlib import Path

# --- BASE DIRECTORIES ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Subdirectories for temporary uploads and extracted board images
UPLOAD_DIR = DATA_DIR / "uploads"
BOARDS_DIR = DATA_DIR / "extracted_boards"

# Ensure required directories exist automatically on startup
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
BOARDS_DIR.mkdir(parents=True, exist_ok=True)

# --- MODEL CONFIGURATIONS ---
# Options: "tiny", "base", "small", "medium", "large"
# "small" offers the best speed/accuracy balance on standard local GPU/CPU
WHISPER_MODEL_NAME = "small"

# Default local LLM for Ollama
DEFAULT_OLLAMA_MODEL = "llama3.2"