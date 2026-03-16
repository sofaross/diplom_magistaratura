import torch
import torch.nn as nn

# ===============================
#  Этот класс объединяет два разных embedding.
# ===============================
class FusionModel(nn.Module):

    def __init__(self, speech_dim=1024, emotion_dim=128, num_classes=8):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(speech_dim + emotion_dim, 256),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, speech_embedding, emotion_embedding):

        combined = torch.cat(
            (speech_embedding, emotion_embedding),
            dim=1
        )

        output = self.classifier(combined)

        return output
