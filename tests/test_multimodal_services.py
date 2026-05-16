from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.audio_io.audio_file_manager import AudioFileManager
from src.services.audio_input_service import AudioInputService
from src.services.multimodal_classification_service import MultimodalClassificationService
from src.services.noise_service import NoiseService
from src.noise.noise_manager import NoiseManager


class FakeSpeechRecognitionService:
    def __init__(self) -> None:
        self.languages: list[str | None] = []

    def recognize(self, audio: np.ndarray, *, language: str | None = None) -> str:
        self.languages.append(language)
        return "test transcription"


class FakeEmotionRecognitionService:
    def recognize(self, audio: np.ndarray) -> tuple[str, dict[str, float]]:
        return "happy", {"happy": 0.9, "sad": 0.1}


class MultimodalServicesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.clean_dir = self.root / "clean"
        self.processed_dir = self.root / "processed"
        self.noisy_dir = self.root / "noisy"
        self.noise_dir = self.root / "noise"

        self.sample_rate = 16000
        clean_manager = AudioFileManager(save_dir=self.clean_dir, sample_rate=self.sample_rate)
        noise_manager = AudioFileManager(save_dir=self.noise_dir, sample_rate=self.sample_rate)

        time = np.arange(self.sample_rate, dtype=np.float32) / float(self.sample_rate)
        clean_audio = 0.1 * np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
        noise_audio = 0.05 * np.sin(2.0 * np.pi * 37.0 * time).astype(np.float32)

        self.clean_file = clean_manager.save(clean_audio, "clean.wav")
        noise_manager.save(noise_audio, "metro.wav")
        noise_manager.save(noise_audio * 0.8, "rain_1.wav")
        noise_manager.save(noise_audio * 0.6, "rain_2.wav")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_audio_input_service_can_prepare_and_persist_audio(self) -> None:
        service = AudioInputService(
            sample_rate=self.sample_rate,
            prepared_file_manager=AudioFileManager(save_dir=self.processed_dir, sample_rate=self.sample_rate),
        )

        payload = service.load_audio_file(self.clean_file, save_prepared_copy=True)

        self.assertEqual(payload.source_type, "file")
        self.assertTrue(payload.original_file_path.exists())
        self.assertIsNotNone(payload.prepared_file_path)
        self.assertTrue(payload.prepared_file_path.exists())
        self.assertGreater(payload.duration_seconds, 0.0)

    def test_multimodal_pipeline_returns_expected_result(self) -> None:
        audio_input_service = AudioInputService(
            sample_rate=self.sample_rate,
            prepared_file_manager=AudioFileManager(save_dir=self.processed_dir, sample_rate=self.sample_rate),
        )
        noise_service = NoiseService(
            noise_manager=NoiseManager(noise_dir=self.noise_dir, sample_rate=self.sample_rate, random_seed=123),
            output_file_manager=AudioFileManager(save_dir=self.noisy_dir, sample_rate=self.sample_rate),
        )
        speech_service = FakeSpeechRecognitionService()
        pipeline = MultimodalClassificationService(
            audio_input_service=audio_input_service,
            noise_service=noise_service,
            speech_recognition_service=speech_service,
            emotion_recognition_service=FakeEmotionRecognitionService(),
        )

        result = pipeline.process_audio_file(
            self.clean_file,
            noise_mode="selected",
            noise_type="rain",
            snr_db=5.0,
            speech_language="ru",
        )

        self.assertEqual(result.recognized_text, "test transcription")
        self.assertEqual(result.recognized_emotion, "happy")
        self.assertEqual(result.noise_mode, "selected")
        self.assertEqual(result.noise_type, "rain")
        self.assertIn(result.noise_variant, {"rain_1", "rain_2"})
        self.assertEqual(speech_service.languages, ["ru"])
        self.assertTrue(Path(result.processed_audio_path).exists())


if __name__ == "__main__":
    unittest.main()
