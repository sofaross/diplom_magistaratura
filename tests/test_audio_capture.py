from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

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

    def test_get_input_device_name_returns_sounddevice_device_name(self) -> None:
        mic = MicrophoneCapture(sample_rate=16000, file_manager=FakeFileManager(), auto_save=False)

        with patch("src.audio_io.audio_capture.sd.query_devices", return_value={"name": "USB Microphone"}):
            device_name = mic.get_input_device_name()

        self.assertEqual(device_name, "USB Microphone")

    def test_get_input_device_name_falls_back_when_query_fails(self) -> None:
        mic = MicrophoneCapture(sample_rate=16000, file_manager=FakeFileManager(), auto_save=False)

        with patch("src.audio_io.audio_capture.sd.query_devices", side_effect=RuntimeError("no device")):
            device_name = mic.get_input_device_name()

        self.assertEqual(device_name, "устройство ввода по умолчанию")

    def test_list_input_devices_returns_only_input_capable_devices(self) -> None:
        mic = MicrophoneCapture(sample_rate=16000, file_manager=FakeFileManager(), auto_save=False, input_device=2)
        fake_devices = [
            {"name": "Speakers", "max_input_channels": 0},
            {"name": "Mic 1", "max_input_channels": 1},
            {"name": "Mic 2", "max_input_channels": 2},
        ]

        with patch("src.audio_io.audio_capture.sd.query_devices", return_value=fake_devices):
            with patch("src.audio_io.audio_capture.sd.default", new=SimpleNamespace(device=(1, 3))):
                devices = mic.list_input_devices()

        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["name"], "Mic 1")
        self.assertTrue(devices[0]["is_default"])
        self.assertEqual(devices[1]["index"], 2)
        self.assertTrue(devices[1]["is_selected"])

    def test_set_input_device_updates_selected_device(self) -> None:
        mic = MicrophoneCapture(sample_rate=16000, file_manager=FakeFileManager(), auto_save=False)

        with patch("src.audio_io.audio_capture.sd.query_devices", return_value={"name": "Mic 3"}):
            mic.set_input_device(3)

        self.assertEqual(mic.input_device, 3)

    def test_listen_passes_selected_input_device_to_sounddevice(self) -> None:
        manager = FakeFileManager(sample_rate=16000)
        mic = MicrophoneCapture(sample_rate=16000, file_manager=manager, auto_save=False, input_device=4)
        fake_audio = np.full((160, 1), 0.1, dtype=np.float32)
        duration = fake_audio.shape[0] / float(mic.sample_rate)

        with patch.object(mic, "check_microphone", return_value=None):
            with patch("src.audio_io.audio_capture.sd.rec", return_value=fake_audio) as mock_rec:
                with patch("src.audio_io.audio_capture.sd.wait", return_value=None):
                    mic.listen(duration=duration)

        self.assertEqual(mock_rec.call_args.kwargs["device"], 4)


if __name__ == "__main__":
    unittest.main()
