import torch
import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2Model

# ===============================
# Класс для извлечения эмбеддингов (умных признаков) из речи с помощью Wav2Vec2.
# ===============================
class SpeechEmbeddingModel:

    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53", device=None):

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        if device is None:
            device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        self.model.eval()

    # ===============================
    # Возвращает размерность эмбеддинга.
    # ===============================
    @property
    def embedding_dim(self):
        return int(self.model.config.hidden_size)

    # ===============================
    # Превращает аудио в эмбеддинг.
    # ===============================
    def extract_embedding(self, audio, sample_rate=16000):

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        elif isinstance(audio, (list, tuple)):
            audio = [
                item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
                for item in audio
            ]

        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32, copy=False)
        elif isinstance(audio, (list, tuple)):
            audio = [np.asarray(item, dtype=np.float32) for item in audio]

        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        hidden_states = outputs.last_hidden_state

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            embedding = torch.mean(hidden_states, dim=1)
        else:
            # Convert input attention mask to feature-vector mask (after conv subsampling).
            feature_mask = self.model._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            feature_mask = feature_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            masked_sum = (hidden_states * feature_mask).sum(dim=1)
            denom = feature_mask.sum(dim=1).clamp(min=1.0)
            embedding = masked_sum / denom

        return embedding
