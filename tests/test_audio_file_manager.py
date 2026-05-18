from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.audio_io.audio_file_manager import AudioFileManager


class AudioFileManagerTests(unittest.TestCase):
    def test_list_audio_files_includes_wav_and_ogg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            wav_path = root / "sample.wav"
            ogg_path = root / "sample.ogg"
            txt_path = root / "sample.txt"

            wav_path.write_bytes(b"wav")
            ogg_path.write_bytes(b"ogg")
            txt_path.write_bytes(b"txt")

            files = AudioFileManager.list_audio_files(root, recursive=False)

        self.assertEqual({path.name for path in files}, {"sample.wav", "sample.ogg"})

    def test_is_supported_audio_file_accepts_ogg(self) -> None:
        self.assertTrue(AudioFileManager.is_supported_audio_file("voice.ogg"))
        self.assertTrue(AudioFileManager.is_supported_audio_file("voice.wav"))
        self.assertFalse(AudioFileManager.is_supported_audio_file("voice.mp3"))


if __name__ == "__main__":
    unittest.main()
