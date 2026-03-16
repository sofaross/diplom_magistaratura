import torch
import torch.nn as nn


class EmotionModel(nn.Module):

    def __init__(self, num_emotions):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64 * 32,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Sequential(

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_emotions)
        )

    def extract_embedding(self, x):

        x = self.cnn(x)

        batch, channels, height, width = x.size()

        x = x.view(batch, width, channels * height)

        lstm_out, _ = self.lstm(x)

        # 128-dim emotion embedding (used for fusion).
        return lstm_out[:, -1, :]

#?
    def forward(self, x, return_embedding=False):

        embedding = self.extract_embedding(x)

        logits = self.fc(embedding)

        if return_embedding:
            return logits, embedding

        return logits
