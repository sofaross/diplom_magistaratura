from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from configs.config import ProjectConfig
from src.services.speech_recognition_service import SpeechRecognitionService


class SpeechRecognitionServiceTests(unittest.TestCase):
    def test_recognize_uses_language_specific_model(self) -> None:
        loaded_model_names: list[str] = []

        def fake_from_pretrained(*, model_name: str, device, sample_rate: int):
            del device, sample_rate
            loaded_model_names.append(model_name)
            return SimpleNamespace(model_name=model_name)

        with patch(
            "src.services.speech_recognition_service.Wav2Vec2Wrapper.from_pretrained",
            side_effect=fake_from_pretrained,
        ):
            with patch(
                "src.services.speech_recognition_service.transcribe",
                side_effect=lambda wrapper, audio, preprocess: f"decoded:{wrapper.model_name}",
            ):
                service = SpeechRecognitionService(config=ProjectConfig(), preprocess=False)
                audio = np.zeros(160, dtype=np.float32)

                ru_text = service.recognize(audio, language="ru")
                en_text = service.recognize(audio, language="english")

        self.assertEqual(
            loaded_model_names,
            [
                ProjectConfig().speech_model_name_ru,
                ProjectConfig().speech_model_name_en,
            ],
        )
        self.assertEqual(ru_text, f"decoded:{ProjectConfig().speech_model_name_ru}")
        self.assertEqual(en_text, f"decoded:{ProjectConfig().speech_model_name_en}")


if __name__ == "__main__":
    unittest.main()
