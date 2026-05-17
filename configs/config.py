from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectConfig:
    repo_root: Path = REPO_ROOT
    sample_rate: int = 16000

    # Datasets
    crema_path: Path = REPO_ROOT / "data" / "raw" / "crema-d" / "AudioWAV"
    ravdess_path: Path = REPO_ROOT / "data" / "raw" / "ravdess"
    noise_dir: Path = REPO_ROOT / "data" / "noise"
    recordings_dir: Path = REPO_ROOT / "data" / "recording"
    clean_recordings_dir: Path = REPO_ROOT / "data" / "recording" / "withoutNoise"
    noisy_recordings_dir: Path = REPO_ROOT / "data" / "recording" / "withNoise"
    processed_audio_dir: Path = REPO_ROOT / "data" / "processed" / "audio"

    # Models
    # Важно: для ASR нужен Wav2Vec2 с CTC головой (ASR-модель), а не просто "xlsr-53" без CTC.
    speech_model_name: str = "facebook/wav2vec2-base-960h"
    default_speech_language: str = "en"
    speech_model_name_en: str = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    speech_model_name_ru: str = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    models_dir: Path = REPO_ROOT / "data" / "processed" / "models"
    emotion_out_dir: Path = REPO_ROOT / "data" / "processed" / "models" / "emotion"
    fusion_out_dir: Path = REPO_ROOT / "data" / "processed" / "models" / "fusion"
    emotion_checkpoint_path: Path = REPO_ROOT / "data" / "processed" / "models" / "emotion" / "emotion_model_final.pt"
    emotion_map_path: Path = REPO_ROOT / "data" / "processed" / "models" / "emotion" / "emotion_map.json"
