from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.audio_io.audio_file_manager import AudioFileManager
from src.noise.noise_manager import NoiseManager
from src.pipeline import MultimodalPipeline


class FakeSpeechRecognizer:
    def __init__(self, text: str = "decoded text") -> None:
        self.text = text
        self.calls: list[np.ndarray] = []

    def transcribe(self, audio):
        self.calls.append(np.asarray(audio, dtype=np.float32))
        return self.text


class FakeEmotionRecognizer:
    def __init__(self, emotion: str = "happy") -> None:
        self.emotion = emotion
        self.calls: list[np.ndarray] = []

    def recognize(self, audio: np.ndarray) -> tuple[str, dict[str, float]]:
        self.calls.append(np.asarray(audio, dtype=np.float32))
        return self.emotion, {self.emotion: 0.9, "sad": 0.1}


class MultimodalPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.noise_dir = self.root / "noise"

        self.sample_rate = 16000
        self.input_manager = AudioFileManager(save_dir=self.input_dir, sample_rate=self.sample_rate)
        self.output_manager = AudioFileManager(save_dir=self.output_dir, sample_rate=self.sample_rate)
        self.noise_file_manager = AudioFileManager(save_dir=self.noise_dir, sample_rate=self.sample_rate)

        time = np.arange(self.sample_rate, dtype=np.float32) / float(self.sample_rate)
        clean_audio = 0.1 * np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
        noise_audio = 0.03 * np.sin(2.0 * np.pi * 50.0 * time).astype(np.float32)

        self.audio_path = self.input_manager.save(clean_audio, "sample.wav")
        self.external_noise_path = self.noise_file_manager.save(noise_audio, "street.wav")
        self.noise_file_manager.save(noise_audio * 0.8, "rain_1.wav")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_pipeline_processes_clean_audio(self) -> None:
        speech = FakeSpeechRecognizer("clean text")
        emotion = FakeEmotionRecognizer("neutral")
        pipeline = MultimodalPipeline(
            audio_manager=self.input_manager,
            processed_audio_manager=self.output_manager,
            noise_manager=NoiseManager(noise_dir=self.noise_dir, sample_rate=self.sample_rate, random_seed=123),
            speech_recognizer=speech,
            emotion_recognizer=emotion,
        )

        result = pipeline.process_audio(self.audio_path)

        self.assertEqual(result.recognized_text, "clean text")
        self.assertEqual(result.predicted_emotion, "neutral")
        self.assertIsNone(result.noise_type)
        self.assertEqual(result.errors, [])
        self.assertEqual(Path(result.processed_audio_path), self.audio_path.resolve())
        self.assertEqual(list(self.output_dir.glob("*.wav")), [])

    def test_pipeline_supports_external_noise_file(self) -> None:
        speech = FakeSpeechRecognizer("noisy text")
        emotion = FakeEmotionRecognizer("angry")
        pipeline = MultimodalPipeline(
            audio_manager=self.input_manager,
            processed_audio_manager=self.output_manager,
            noise_manager=NoiseManager(noise_dir=self.noise_dir, sample_rate=self.sample_rate, random_seed=123),
            speech_recognizer=speech,
            emotion_recognizer=emotion,
        )

        result = pipeline.process_audio(
            self.audio_path,
            noise_file=self.external_noise_path,
            snr_db=5.0,
        )

        self.assertEqual(result.recognized_text, "noisy text")
        self.assertEqual(result.predicted_emotion, "angry")
        self.assertEqual(result.noise_type, "street")
        self.assertEqual(result.snr_db, 5.0)
        self.assertEqual(result.errors, [])
        self.assertTrue(Path(result.processed_audio_path).exists())

    def test_pipeline_supports_random_noise_mode(self) -> None:
        speech = FakeSpeechRecognizer("random text")
        emotion = FakeEmotionRecognizer("fear")
        pipeline = MultimodalPipeline(
            audio_manager=self.input_manager,
            processed_audio_manager=self.output_manager,
            noise_manager=NoiseManager(noise_dir=self.noise_dir, sample_rate=self.sample_rate, random_seed=123),
            speech_recognizer=speech,
            emotion_recognizer=emotion,
        )

        result = pipeline.process_audio(
            self.audio_path,
            use_random_noise=True,
            snr_db=7.0,
        )

        self.assertEqual(result.recognized_text, "random text")
        self.assertEqual(result.predicted_emotion, "fear")
        self.assertIsNotNone(result.noise_type)
        self.assertEqual(result.snr_db, 7.0)
        self.assertEqual(result.errors, [])
        self.assertTrue(Path(result.processed_audio_path).exists())


if __name__ == "__main__":
    unittest.main()
