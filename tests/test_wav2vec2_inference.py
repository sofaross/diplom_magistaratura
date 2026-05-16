from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.inference.wav2vec2_inference import (
    extract_embedding,
    prepare_audio_batch,
    transcribe,
    transcribe_and_embed,
)
from src.models.speech_model import Wav2Vec2Multimodal
from src.models.wav2vec2_wrapper import Wav2Vec2Wrapper


class FakeProcessor:
    """Минимальный processor для smoke-тестов без HuggingFace."""

    def __call__(
        self,
        batch,
        *,
        sampling_rate,
        return_tensors,
        padding,
        return_attention_mask,
    ):
        del sampling_rate, return_tensors, padding, return_attention_mask

        max_len = max(len(item) for item in batch)
        input_values = []
        attention_mask = []

        for item in batch:
            audio = np.asarray(item, dtype=np.float32)
            pad_width = max_len - int(audio.shape[0])
            input_values.append(np.pad(audio, (0, pad_width), constant_values=0.0))
            attention_mask.append(np.pad(np.ones_like(audio, dtype=np.int64), (0, pad_width), constant_values=0))

        return {
            "input_values": torch.tensor(np.stack(input_values), dtype=torch.float32),
            "attention_mask": torch.tensor(np.stack(attention_mask), dtype=torch.long),
        }

    def batch_decode(self, pred_ids: torch.Tensor) -> list[str]:
        return [" ".join(str(int(token)) for token in row.tolist()) for row in pred_ids]


class FakeBaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)

    def _get_feature_vector_attention_mask(self, feature_length: int, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=torch.bool)
        current_length = int(mask.shape[1])
        if current_length == feature_length:
            return mask
        if current_length > feature_length:
            return mask[:, :feature_length]

        pad = torch.zeros((mask.shape[0], feature_length - current_length), dtype=torch.bool, device=mask.device)
        return torch.cat([mask, pad], dim=1)

    def forward(
        self,
        input_values: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        del attention_mask, return_dict

        last_hidden_state = torch.stack(
            (
                input_values,
                input_values + 1.0,
                input_values + 2.0,
                input_values + 3.0,
            ),
            dim=-1,
        )
        hidden_states = None
        if output_hidden_states:
            hidden_states = (
                last_hidden_state - 10.0,
                last_hidden_state,
                last_hidden_state + 10.0,
            )
        return SimpleNamespace(last_hidden_state=last_hidden_state, hidden_states=hidden_states)


class FakeCTCModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wav2vec2 = FakeBaseModel()
        self.config = SimpleNamespace(hidden_size=4)
        self.dropout = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(4, 3, bias=False)

        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        base_outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = self.dropout(base_outputs.last_hidden_state)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)


def build_wrapper() -> Wav2Vec2Wrapper:
    return Wav2Vec2Wrapper(
        model_name="fake-wav2vec2",
        device=torch.device("cpu"),
        processor=FakeProcessor(),
        model=FakeCTCModel(),
        sample_rate=16000,
    )


class Wav2Vec2InferenceTests(unittest.TestCase):
    def test_prepare_audio_batch_pads_and_builds_mask(self) -> None:
        wrapper = build_wrapper()
        audio = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0], dtype=np.float32),
        ]

        input_values, attention_mask = prepare_audio_batch(wrapper, audio, preprocess=False)

        expected_values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]], dtype=torch.float32)
        expected_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long)

        torch.testing.assert_close(input_values.cpu(), expected_values)
        torch.testing.assert_close(attention_mask.cpu(), expected_mask)

    def test_extract_embedding_supports_mean_max_and_intermediate_layer(self) -> None:
        wrapper = build_wrapper()
        audio = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0], dtype=np.float32),
        ]

        mean_embedding = extract_embedding(wrapper, audio, preprocess=False)
        max_embedding = extract_embedding(wrapper, audio, pool="max", preprocess=False)
        layer_zero_embedding = extract_embedding(wrapper, audio, layer=0, preprocess=False)

        torch.testing.assert_close(
            mean_embedding.cpu(),
            torch.tensor([[2.0, 3.0, 4.0, 5.0], [4.0, 5.0, 6.0, 7.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            max_embedding.cpu(),
            torch.tensor([[3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            layer_zero_embedding.cpu(),
            torch.tensor([[-8.0, -7.0, -6.0, -5.0], [-6.0, -5.0, -4.0, -3.0]], dtype=torch.float32),
        )

    def test_transcribe_and_embed_matches_separate_calls(self) -> None:
        wrapper = build_wrapper()
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        text = transcribe(wrapper, audio, preprocess=False)
        embedding = extract_embedding(wrapper, audio, preprocess=False)
        combined_text, combined_embedding = transcribe_and_embed(wrapper, audio, preprocess=False)

        self.assertEqual(text, combined_text)
        torch.testing.assert_close(embedding.cpu(), combined_embedding.cpu())

    def test_legacy_wrapper_delegates_to_new_api(self) -> None:
        wrapper = build_wrapper()
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with patch("src.models.speech_model.Wav2Vec2Wrapper.from_pretrained", return_value=wrapper):
            legacy = Wav2Vec2Multimodal(preprocess=False, audio_processor=None, strict=True)

        legacy_text = legacy.transcribe(audio)
        legacy_embedding = legacy.extract_embedding(audio)
        legacy_combined_text, legacy_combined_embedding = legacy.transcribe_and_embed(audio)
        legacy_process = legacy.process(audio, return_embeddings=True)

        self.assertEqual(legacy.embedding_dim, wrapper.hidden_size)
        self.assertEqual(legacy_text, legacy_combined_text)
        self.assertEqual(legacy_text, legacy_process["text"])
        torch.testing.assert_close(legacy_embedding.cpu(), legacy_combined_embedding.cpu())
        torch.testing.assert_close(legacy_embedding.cpu(), legacy_process["embedding"].cpu())


if __name__ == "__main__":
    unittest.main()
