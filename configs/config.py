from dataclasses import dataclass
from pathlib import Path

from configs.model_runtime import (
    ASR_MODELS_DIR,
    DEFAULT_SPEECH_MODEL,
    DEFAULT_SPEECH_MODEL_EN,
    DEFAULT_SPEECH_MODEL_RU,
    DEFAULT_TEXT_CORRECTION_MODEL_EN,
    DEFAULT_TEXT_CORRECTION_MODEL_RU,
    LOCAL_FILES_ONLY,
    OFFLINE_MODE,
    TEXT_CORRECTION_MODELS_DIR,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectConfig:
    repo_root: Path = REPO_ROOT
    sample_rate: int = 16000

    # Datasets
    crema_path: Path = REPO_ROOT / "data" / "raw" / "crema-d" / "AudioWAV"
    ravdess_path: Path = REPO_ROOT / "data" / "raw" / "ravdess"
    noise_dir: Path = REPO_ROOT / "data" / "noise"
    test_data_dir: Path = REPO_ROOT / "data" / "test"
    recordings_dir: Path = REPO_ROOT / "data" / "recording"
    clean_recordings_dir: Path = recordings_dir / "real" / "withoutNoise"
    noisy_recordings_dir: Path = recordings_dir / "real" / "withNoise"
    processed_audio_dir: Path = test_data_dir / "audio"

    # Models
    # Важно: для ASR нужен Wav2Vec2 с CTC головой (ASR-модель), а не просто "xlsr-53" без CTC.
    offline_mode: bool = OFFLINE_MODE
    local_files_only: bool = LOCAL_FILES_ONLY
    asr_models_dir: Path = ASR_MODELS_DIR
    speech_model_name: str = DEFAULT_SPEECH_MODEL
    default_speech_language: str = "en"
    speech_model_name_en: str = DEFAULT_SPEECH_MODEL_EN
    speech_model_name_ru: str = DEFAULT_SPEECH_MODEL_RU
    text_correction_models_dir: Path = TEXT_CORRECTION_MODELS_DIR
    text_correction_model_name_en: str = DEFAULT_TEXT_CORRECTION_MODEL_EN
    text_correction_model_name_ru: str = DEFAULT_TEXT_CORRECTION_MODEL_RU
    models_dir: Path = REPO_ROOT / "data" / "processed" / "models"
    emotion_out_dir: Path = REPO_ROOT / "data" / "processed" / "models" / "emotion"
    fusion_out_dir: Path = REPO_ROOT / "data" / "processed" / "models" / "fusion"
    emotion_checkpoint_path: Path = REPO_ROOT / "data" / "processed" / "models" / "emotion" / "emotion_model_final.pt"
    emotion_map_path: Path = REPO_ROOT / "data" / "processed" / "models" / "emotion" / "emotion_map.json"
