from __future__ import annotations

from pathlib import Path

from src.dto.multimodal_result_dto import PreparedAudioDto, ResultDto
from src.services.audio_input_service import AudioInputService
from src.services.emotion_recognition_service import EmotionRecognitionService
from src.services.noise_service import NoiseService
from src.services.speech_recognition_service import SpeechRecognitionService


class MultimodalClassificationService:
    """Общий pipeline: аудио -> шум -> ASR -> emotion -> итоговый результат."""

    def __init__(
        self,
        *,
        audio_input_service: AudioInputService | None = None,
        noise_service: NoiseService | None = None,
        speech_recognition_service: SpeechRecognitionService | None = None,
        emotion_recognition_service: EmotionRecognitionService | None = None,
    ) -> None:
        self.audio_input_service = audio_input_service or AudioInputService()
        self.noise_service = noise_service or NoiseService()
        self.speech_recognition_service = speech_recognition_service or SpeechRecognitionService()
        self.emotion_recognition_service = emotion_recognition_service or EmotionRecognitionService()

    def process_audio_file(
        self,
        file_path: str | Path,
        *,
        noise_mode: str = "none",
        noise_type: str | None = None,
        snr_db: float | None = None,
        speech_language: str | None = None,
    ) -> ResultDto:
        payload = self.audio_input_service.load_audio_file(file_path, save_prepared_copy=False)
        return self._run_pipeline(
            payload,
            noise_mode=noise_mode,
            noise_type=noise_type,
            snr_db=snr_db,
            speech_language=speech_language,
        )

    def process_microphone_input(
        self,
        *,
        duration: float = 5.0,
        filename: str | None = None,
        noise_mode: str = "none",
        noise_type: str | None = None,
        snr_db: float | None = None,
        speech_language: str | None = None,
    ) -> ResultDto:
        payload = self.audio_input_service.capture_voice_message(
            duration=duration,
            filename=filename,
            save_prepared_copy=False,
        )
        return self._run_pipeline(
            payload,
            noise_mode=noise_mode,
            noise_type=noise_type,
            snr_db=snr_db,
            speech_language=speech_language,
        )

    def _run_pipeline(
        self,
        payload: PreparedAudioDto,
        *,
        noise_mode: str,
        noise_type: str | None,
        snr_db: float | None,
        speech_language: str | None,
    ) -> ResultDto:
        normalized_mode = str(noise_mode).strip().lower()

        if normalized_mode == "none":
            payload = self.audio_input_service.persist_prepared_audio(payload)
            noise_metadata = {
                "noise_applied": False,
                "noise_mode": "none",
                "noise_type": None,
                "noise_variant": None,
                "snr_db": None,
            }
        elif normalized_mode == "random":
            payload, noise_metadata = self.noise_service.apply_random_noise(
                payload,
                snr_db=float(10.0 if snr_db is None else snr_db),
            )
        elif normalized_mode == "selected":
            if not noise_type:
                raise ValueError("noise_type must be provided when noise_mode='selected'.")
            payload, noise_metadata = self.noise_service.apply_selected_noise(
                payload,
                noise_type=noise_type,
                snr_db=float(10.0 if snr_db is None else snr_db),
            )
        else:
            raise ValueError("noise_mode must be one of: 'none', 'random', 'selected'.")

        text = self.speech_recognition_service.recognize(payload.audio, language=speech_language)
        emotion, probabilities = self.emotion_recognition_service.recognize(payload.audio)

        processed_path = payload.prepared_file_path or payload.original_file_path
        return ResultDto(
            source_type=payload.source_type,
            original_audio_path=str(payload.original_file_path),
            processed_audio_path=str(processed_path),
            sample_rate=payload.sample_rate,
            duration_seconds=payload.duration_seconds,
            noise_applied=bool(noise_metadata["noise_applied"]),
            noise_mode=str(noise_metadata["noise_mode"]),
            noise_type=noise_metadata["noise_type"],
            noise_variant=noise_metadata["noise_variant"],
            snr_db=noise_metadata["snr_db"],
            recognized_text=text,
            recognized_emotion=emotion,
            emotion_probabilities=probabilities,
        )


__all__ = ["MultimodalClassificationService"]
