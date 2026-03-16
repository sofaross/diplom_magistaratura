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

    def extract_embedding(self, x, lengths=None):
        # lengths здесь игнорируем (это baseline-модель), но параметр оставляем
        # для совместимости с улучшенным пайплайном (collate может отдавать lengths).

        x = self.cnn(x)

        batch, channels, height, width = x.size()

        x = x.view(batch, width, channels * height)

        lstm_out, _ = self.lstm(x)

        # 128-dim emotion embedding (used for fusion).
        return lstm_out[:, -1, :]

    def forward(self, x, lengths=None, return_embedding=False):

        embedding = self.extract_embedding(x, lengths=lengths)

        logits = self.fc(embedding)

        if return_embedding:
            return logits, embedding

        return logits


class AttentionPooling(nn.Module):
    """Attention pooling по времени с маскированием padding."""

    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D]
        scores = self.net(x).squeeze(-1)  # [B, T]

        if lengths is not None:
            # positions >= length -> mask
            bsz, t = scores.shape
            device = scores.device
            mask = torch.arange(t, device=device).unsqueeze(0).expand(bsz, t) >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)

        attn = torch.softmax(scores, dim=1)  # [B, T]
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # [B, D]
        return pooled


class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling по времени (mean+std) с маскированием padding.

    Часто даёт лучшее качество, чем простой attention-mean, потому что сохраняет
    не только "средний" контекст, но и вариативность/динамику (std) по времени.
    """

    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.net = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D]
        scores = self.net(x).squeeze(-1)  # [B, T]

        if lengths is not None:
            bsz, t = scores.shape
            device = scores.device
            mask = torch.arange(t, device=device).unsqueeze(0).expand(bsz, t) >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)

        attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]

        mean = torch.sum(attn * x, dim=1)  # [B, D]
        var = torch.sum(attn * (x - mean.unsqueeze(1)) ** 2, dim=1)  # [B, D]
        std = torch.sqrt(torch.clamp(var, min=self.eps))

        return torch.cat([mean, std], dim=1)  # [B, 2D]


class _ResConvBlock(nn.Module):
    """Residual-блок для 2D CNN на mel-спектрограммах."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        dropout: float = 0.1,
        pool: tuple[int, int] | None = (2, 2),
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU()

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.drop = nn.Dropout2d(float(dropout))
        self.pool = nn.MaxPool2d(pool) if pool is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.skip is not None:
            identity = self.skip(identity)

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + identity)
        out = self.drop(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


class EmotionModelImproved(nn.Module):
    """Улучшенная модель эмоций: CNN (res-блоки) + BiLSTM + attentive stats pooling.

    Ключевые улучшения относительно baseline:
    - Residual CNN-блоки + BatchNorm/Dropout (модель глубже и устойчивее).
    - BiLSTM вместо LSTM (контекст слева/справа).
    - Attentive statistics pooling (mean+std) вместо "последнего таймстепа".
    - Поддержка lengths, чтобы padding не ломал представление.

    Важно: выходной `embedding_dim` оставляем 128, чтобы модель была совместима
    с fusion-частью проекта (emotion embedding в FusionModel по умолчанию 128).
    """

    def __init__(
        self,
        num_emotions: int,
        *,
        embedding_dim: int = 128,
        rnn_hidden_size: int = 128,
        rnn_layers: int = 2,
        cnn_dropout: float = 0.15,
        rnn_dropout: float = 0.2,
        head_dropout: float = 0.35,
    ):
        super().__init__()

        # Вход: [B, 1, 128, T]
        self.cnn = nn.Sequential(
            _ResConvBlock(1, 32, dropout=float(cnn_dropout), pool=(2, 2)),    # 128 -> 64,  T -> T/2
            _ResConvBlock(32, 64, dropout=float(cnn_dropout), pool=(2, 2)),   # 64  -> 32,  T -> T/4
            _ResConvBlock(64, 128, dropout=float(cnn_dropout), pool=(2, 2)),  # 32  -> 16,  T -> T/8
            _ResConvBlock(128, 256, dropout=float(cnn_dropout), pool=(2, 1)), # 16  -> 8,   T unchanged
        )

        # После CNN: [B, 256, 8, T/8] => последовательность [B, T/8, 2048]
        self._time_downsample = 8
        self._cnn_out_dim = 256 * 8

        # Проекция, чтобы RNN работал быстрее и стабильнее.
        self._rnn_in = 256
        self.pre_rnn = nn.Sequential(
            nn.Linear(self._cnn_out_dim, self._rnn_in),
            nn.LayerNorm(self._rnn_in),
            nn.SiLU(),
            nn.Dropout(float(head_dropout)),
        )

        self.rnn = nn.LSTM(
            input_size=self._rnn_in,
            hidden_size=int(rnn_hidden_size),
            num_layers=int(rnn_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(rnn_dropout) if int(rnn_layers) > 1 else 0.0,
        )

        self._rnn_out_dim = 2 * int(rnn_hidden_size)  # bi-directional
        self.pool = AttentiveStatsPooling(self._rnn_out_dim, attn_dim=128, dropout=float(head_dropout))
        self._pooled_dim = 2 * self._rnn_out_dim  # mean+std

        # Финальный 128-dim emotion embedding (для fusion и классификации).
        self.embedding_dim = int(embedding_dim)
        self.emb_proj = nn.Sequential(
            nn.LayerNorm(self._pooled_dim),
            nn.Linear(self._pooled_dim, self.embedding_dim),
            nn.SiLU(),
            nn.Dropout(float(head_dropout)),
        )

        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.SiLU(),
            nn.Dropout(float(head_dropout)),
            nn.Linear(128, int(num_emotions)),
        )

    def _downsample_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        # MaxPool2d(2,2) три раза уменьшает длину по времени в 8 раз.
        out = lengths // int(self._time_downsample)
        return torch.clamp(out, min=1)

    def extract_embedding(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.cnn(x)

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # [B, T, C*F]
        x = self.pre_rnn(x)  # [B, T, rnn_in]

        rnn_lengths = None
        if lengths is not None:
            rnn_lengths = self._downsample_lengths(lengths.to(x.device))

        if rnn_lengths is not None:
            # Ускоряем RNN и убираем влияние padding.
            packed = nn.utils.rnn.pack_padded_sequence(
                x, rnn_lengths.detach().to("cpu"), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            rnn_out, _ = self.rnn(x)

        pooled = self.pool(rnn_out, lengths=rnn_lengths)
        emb = self.emb_proj(pooled)
        return emb

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None, return_embedding: bool = False):
        emb = self.extract_embedding(x, lengths=lengths)
        logits = self.head(emb)

        if return_embedding:
            return logits, emb

        return logits
