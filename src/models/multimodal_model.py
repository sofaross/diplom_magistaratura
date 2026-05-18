import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """Мультимодальная fusion-модель на проекциях и gated fusion.

    Логика такая:
    1. Приводим speech и emotion embedding к общей размерности.
    2. Строим gate, который по каждому признаку решает, чему доверять сильнее:
       speech-ветке или emotion-ветке.
    3. На классификацию подаём исходные проекции обеих модальностей и их gated-смешивание.
    """

    def __init__(
        self,
        speech_dim: int = 1024,
        emotion_dim: int = 128,
        num_classes: int = 8,
        *,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        modality_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.modality_dropout_prob = float(modality_dropout_prob)

        self.speech_projection = nn.Sequential(
            nn.LayerNorm(int(speech_dim)),
            nn.Linear(int(speech_dim), int(projection_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )

        self.emotion_projection = nn.Sequential(
            nn.LayerNorm(int(emotion_dim)),
            nn.Linear(int(emotion_dim), int(projection_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )

        self.gate = nn.Sequential(
            nn.Linear(int(projection_dim) * 2, int(projection_dim)),
            nn.Sigmoid(),
        )

        classifier_input_dim = int(projection_dim) * 3
        reduced_hidden_dim = max(int(hidden_dim) // 2, 64)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), reduced_hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(reduced_hidden_dim, int(num_classes)),
        )

    def _apply_modality_dropout(
        self,
        speech_features: torch.Tensor,
        emotion_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout_prob <= 0:
            return speech_features, emotion_features

        probs = torch.rand(speech_features.shape[0], device=speech_features.device)
        drop_speech = probs < (self.modality_dropout_prob / 2.0)
        drop_emotion = (probs >= (self.modality_dropout_prob / 2.0)) & (probs < self.modality_dropout_prob)

        if drop_speech.any():
            speech_features = speech_features.clone()
            speech_features[drop_speech] = 0.0
        if drop_emotion.any():
            emotion_features = emotion_features.clone()
            emotion_features[drop_emotion] = 0.0

        return speech_features, emotion_features

    def forward(self, speech_embedding: torch.Tensor, emotion_embedding: torch.Tensor) -> torch.Tensor:
        speech_features = self.speech_projection(speech_embedding)
        emotion_features = self.emotion_projection(emotion_embedding)
        speech_features, emotion_features = self._apply_modality_dropout(speech_features, emotion_features)

        gate = self.gate(torch.cat((speech_features, emotion_features), dim=1))
        fused_features = gate * speech_features + (1.0 - gate) * emotion_features

        classifier_input = torch.cat((speech_features, emotion_features, fused_features), dim=1)
        return self.classifier(classifier_input)
