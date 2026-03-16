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

    # Models
    speech_model_name: str = "facebook/wav2vec2-large-xlsr-53"
    models_dir: Path = REPO_ROOT / "data" / "processed" / "models"
    emotion_out_dir: Path = REPO_ROOT / "data" / "processed" / "models" / "emotion"
    fusion_out_dir: Path = REPO_ROOT / "data" / "processed" / "models" / "fusion"
