from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class PreparedAudioDto:
    source_type: str
    original_file_path: Path
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    prepared_file_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ResultDto:
    source_type: str
    original_audio_path: str
    processed_audio_path: str
    sample_rate: int
    duration_seconds: float
    noise_applied: bool
    noise_mode: str
    noise_type: str | None
    noise_variant: str | None
    snr_db: float | None
    recognized_text: str
    recognized_emotion: str
    emotion_probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ResponseDto = ResultDto


__all__ = ["PreparedAudioDto", "ResultDto", "ResponseDto"]
