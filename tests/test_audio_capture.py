from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.audio_io.audio_capture import MicrophoneCapture


class FakeFileManager:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = int(sample_rate)
        self.saved: list[tuple[np.ndarray, str | None]] = []

    def save(self, audio: np.ndarray, filename: str | None = None) -> Path:
        self.saved.append((np.asarray(audio, dtype=np.float32).copy(), filename))
        return Path("fake_recording.wav")


class AudioCaptureTests(unittest.TestCase):
    def _run_capture(self, mic: MicrophoneCapture, num_frames: int = 160) -> np.ndarray:
        fake_audio = np.full((num_frames, 1), 0.1, dtype=np.float32)
        duration = num_frames / float(mic.sample_rate)

        with patch.object(mic, "check_microphone", return_value=None):
            with patch("src.audio_io.audio_capture.sd.rec", return_value=fake_audio):
                with patch("src.audio_io.audio_capture.sd.wait", return_value=None):
                    return mic.listen(duration=duration)

    def test_auto_save_false_does_not_save_when_save_not_specified(self) -> None:
        manager = FakeFileManager(sample_rate=16000)
        mic = MicrophoneCapture(sample_rate=16000, file_manager=manager, auto_save=False)

        audio = self._run_capture(mic)

        self.assertEqual(audio.shape[0], 160)
        self.assertEqual(manager.saved, [])
        self.assertIsNone(mic.last_saved_path)

    def test_listen_and_save_overrides_auto_save_false(self) -> None:
        manager = FakeFileManager(sample_rate=16000)
        mic = MicrophoneCapture(sample_rate=16000, file_manager=manager, auto_save=False)
        fake_audio = np.full((160, 1), 0.1, dtype=np.float32)
        duration = fake_audio.shape[0] / float(mic.sample_rate)

        with patch.object(mic, "check_microphone", return_value=None):
            with patch("src.audio_io.audio_capture.sd.rec", return_value=fake_audio):
                with patch("src.audio_io.audio_capture.sd.wait", return_value=None):
                    audio, saved_path = mic.listen_and_save(duration=duration, filename="sample.wav")

        self.assertEqual(audio.shape[0], 160)
        self.assertEqual(saved_path, Path("fake_recording.wav"))
        self.assertEqual(len(manager.saved), 1)
        self.assertEqual(manager.saved[0][1], "sample.wav")


if __name__ == "__main__":
    unittest.main()
