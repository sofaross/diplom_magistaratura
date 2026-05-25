from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Центральные настройки speech/text-моделей.
# Если OFFLINE_MODE=True, проект по умолчанию использует только локальные каталоги моделей.
OFFLINE_MODE: bool = True
LOCAL_FILES_ONLY: bool = OFFLINE_MODE

# Каталоги локальных моделей
ASR_MODELS_DIR: Path = REPO_ROOT / "data" / "processed" / "models" / "asr"
TEXT_CORRECTION_MODELS_DIR: Path = REPO_ROOT / "data" / "processed" / "models" / "text_correction"

# Локальные пути
LOCAL_ASR_MODEL_EN: Path = ASR_MODELS_DIR / "en"
LOCAL_ASR_MODEL_RU: Path = ASR_MODELS_DIR / "ru"
LOCAL_TEXT_CORRECTION_MODEL_EN: Path = TEXT_CORRECTION_MODELS_DIR / "en"
LOCAL_TEXT_CORRECTION_MODEL_RU: Path = TEXT_CORRECTION_MODELS_DIR / "ru"

# Hugging Face identifiers
REMOTE_ASR_MODEL_EN: str = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
REMOTE_ASR_MODEL_RU: str = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
REMOTE_TEXT_CORRECTION_MODEL_EN: str = "dayyanj/dj-ai-asr-grammar-corrector-small"
REMOTE_TEXT_CORRECTION_MODEL_RU: str = "ai-forever/sage-fredt5-distilled-95m"


def _pick_reference(local_path: Path, remote_name: str) -> str:
    return str(local_path if OFFLINE_MODE else remote_name)


DEFAULT_SPEECH_MODEL: str = _pick_reference(LOCAL_ASR_MODEL_EN, REMOTE_ASR_MODEL_EN)
DEFAULT_SPEECH_MODEL_EN: str = _pick_reference(LOCAL_ASR_MODEL_EN, REMOTE_ASR_MODEL_EN)
DEFAULT_SPEECH_MODEL_RU: str = _pick_reference(LOCAL_ASR_MODEL_RU, REMOTE_ASR_MODEL_RU)
DEFAULT_TEXT_CORRECTION_MODEL_EN: str = _pick_reference(
    LOCAL_TEXT_CORRECTION_MODEL_EN,
    REMOTE_TEXT_CORRECTION_MODEL_EN,
)
DEFAULT_TEXT_CORRECTION_MODEL_RU: str = _pick_reference(
    LOCAL_TEXT_CORRECTION_MODEL_RU,
    REMOTE_TEXT_CORRECTION_MODEL_RU,
)
