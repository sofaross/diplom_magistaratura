from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from configs.config import ProjectConfig
from src.constants.emotions import CREMA_EMOTIONS, RAVDESS_EMOTIONS
from src.features.feature_extraction import extract_mel_spectrogram
from src.noise.noise_manager import NoiseManager
from src.preprocessing.audio_processing import load_audio, normalize_audio, trim_silence

DEFAULT_DATASET_CONFIG = ProjectConfig()


# ===============================
# Загрудаем CREMA-D датасет
# ===============================

def load_crema_d(dataset_path):

    data = []

    dataset_path = Path(dataset_path)

    for file in dataset_path.glob("**/*.wav"):

        filename = file.name

        parts = filename.split("_")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        emotion = CREMA_EMOTIONS.get(emotion_code)

        if emotion is None:
            continue

        data.append({
            "path": str(file),
            "emotion": emotion,
        })

    df = pd.DataFrame(data, columns=["path", "emotion"])

    return df


# ===============================
# Загружем RAVDESS датасет
# ===============================

def load_ravdess(dataset_path):

    data = []

    dataset_path = Path(dataset_path)

    for file in dataset_path.glob("**/*.wav"):

        filename = file.name

        parts = filename.split("-")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        emotion = RAVDESS_EMOTIONS.get(emotion_code)

        if emotion is None:
            continue

        data.append({
            "path": str(file),
            "emotion": emotion,
        })

    df = pd.DataFrame(data, columns=["path", "emotion"])

    return df

# ===============================
# Кодирование эмоций
# ===============================

def encode_labels(df, emotion_to_id=None):

    if not isinstance(df, pd.DataFrame):
        raise ValueError("encode_labels: ожидается pandas.DataFrame")

    if "emotion" not in df.columns:
        raise ValueError("encode_labels: в датафрейме нет колонки 'emotion' (проверьте загрузку датасета)")

    if len(df) == 0:
        raise ValueError(
            "encode_labels: датасет пустой (0 примеров). "
            "Проверьте пути к датасетам и наличие .wav файлов."
        )

    if emotion_to_id is None:
        emotions = sorted(df["emotion"].unique())
        emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotions)}

    df = df.copy()
    df["label"] = df["emotion"].map(emotion_to_id)

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df, emotion_to_id


# ===============================
# Разделение датасета (для обучения (72% данных), для проверки во время обучения (8% данных), для финального экзамена (20% данных))
# ===============================

def split_dataset(df):

    if len(df) == 0:
        raise ValueError("split_dataset: датасет пустой (0 примеров)")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["label"],
        random_state=42
    )

    return train_df, val_df, test_df


# ===============================
# Подготовка датасета
# ===============================

def prepare_splits(crema_path, ravdess_path,
    emotion_map=None,
    verbose=True,
    emotion_set: int = 6,
):

    if verbose:
        print("Загружаем наборы данных...")

    crema_df = load_crema_d(crema_path)
    ravdess_df = load_ravdess(ravdess_path)

    if verbose:
        print(f"CREMA-D примеров: {len(crema_df)}")
        print(f"RAVDESS примеров: {len(ravdess_df)}")

    df = pd.concat([crema_df, ravdess_df], ignore_index=True)

    if emotion_set == 6:
        allowed = set(CREMA_EMOTIONS.values())
        df = df[df["emotion"].isin(allowed)].reset_index(drop=True)
    elif emotion_set == 8:
        pass
    else:
        raise ValueError("emotion_set должен быть 6 или 8")

    if verbose:
        print("Всего примеров:", len(df))

    if len(df) == 0:
        raise ValueError(
            "Не найдено ни одного .wav файла для обучения эмоций. "
            "Проверьте пути:\n"
            f"  CREMA-D: {crema_path}\n"
            f"  RAVDESS: {ravdess_path}"
        )

    df, emotion_map = encode_labels(df, emotion_to_id=emotion_map)

    train_df, val_df, test_df = split_dataset(df)

    if verbose:
        print("Размер обучения:", len(train_df))
        print("Размер проверки:", len(val_df))
        print("Размер экзамена:", len(test_df))

    return train_df, val_df, test_df, emotion_map


# ===============================
# Класс для подготовки данных для нейросети(на входе папка с записями, на выходе - пары (звук, эмоция))
# ===============================

@dataclass(frozen=True)
class MelAugmentConfig:
    # Вероятности и диапазоны аугментаций (работаем в пространстве мел-спектрограмм).
    # Важно: это быстрые мел-доменные аппроксимации, чтобы 30–40 эпох на CPU были выполнимы.

    # 1) Шум
    noise_prob: float = 0.5
    noise_std_min: float = 0.0
    noise_std_max: float = 0.25

    # 2) Растяжение по времени (меняем длину по оси времени)
    time_stretch_prob: float = 0.35
    time_stretch_min: float = 0.85
    time_stretch_max: float = 1.20

    # 3) Сдвиг "высоты тона" (в мел-домене: смещение по оси частот)
    pitch_shift_prob: float = 0.35
    pitch_shift_bins: int = 6

    # Дополнительно: SpecAugment
    time_mask_prob: float = 0.35
    time_mask_param: int = 30
    freq_mask_prob: float = 0.35
    freq_mask_param: int = 12

    # Дополнительно: сдвиг по времени (эмулируем неточное выравнивание/задержку)
    time_shift_prob: float = 0.25
    # Максимальный сдвиг как доля длины (например, 0.1 = до 10% фреймов влево/вправо)
    time_shift_max_frac: float = 0.10


@dataclass(frozen=True)
class WaveformNoiseAugmentConfig:
    """Параметры шумовой аугментации в waveform-домене до построения mel."""

    noise_prob: float = 0.5
    noise_dir: Path = DEFAULT_DATASET_CONFIG.noise_dir
    noise_types: tuple[str, ...] = ("white", "pink", "brown", "real")
    snr_min: float = 5.0
    snr_max: float = 20.0
    sample_rate: int = DEFAULT_DATASET_CONFIG.sample_rate
    random_seed: int | None = None


def _standardize_mel(mel: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Нормализация мел-спектрограммы по одному примеру (mean=0, std=1).

    Это стабилизирует обучение и делает паддинг значением 0 менее вредным.
    """

    mean = mel.mean()
    std = mel.std(unbiased=False)
    if not torch.isfinite(std) or float(std) < eps:
        return mel - mean
    return (mel - mean) / (std + eps)


def _mel_add_noise(mel: torch.Tensor, std: float) -> torch.Tensor:
    return mel + torch.randn_like(mel) * float(std)


def _mel_time_stretch(mel: torch.Tensor, factor: float) -> torch.Tensor:
    """Меняет длительность мел-спектрограммы по времени через интерполяцию."""

    factor = float(factor)
    if factor <= 0:
        return mel

    _, n_mels, n_frames = mel.shape
    new_frames = max(1, int(round(n_frames * factor)))
    if new_frames == n_frames:
        return mel

    # F.interpolate работает с 4D: [N, C, H, W]. Тут H=n_mels, W=time.
    x = mel.unsqueeze(0)  # [1, 1, n_mels, T]
    x = F.interpolate(x, size=(n_mels, new_frames), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def _mel_pitch_shift(mel: torch.Tensor, shift_bins: int) -> torch.Tensor:
    """Сдвиг по оси частот (mel bins). Это быстрая аппроксимация pitch shift."""

    if shift_bins == 0:
        return mel

    _, n_mels, n_frames = mel.shape
    out = torch.zeros_like(mel)

    if shift_bins > 0:
        # Поднимаем "высоту": переносим энергию в более высокие mel bins.
        if shift_bins >= n_mels:
            return out
        out[:, shift_bins:, :] = mel[:, : n_mels - shift_bins, :]
    else:
        # Опускаем "высоту".
        k = -shift_bins
        if k >= n_mels:
            return out
        out[:, : n_mels - k, :] = mel[:, k:, :]

    return out


def _mel_time_shift(mel: torch.Tensor, shift_frames: int) -> torch.Tensor:
    """Сдвиг по времени с заполнением нулями (не циклический).

    Полезно как аугментация: модель меньше цепляется за абсолютную позицию
    начала/конца фразы в окне.
    """

    if shift_frames == 0:
        return mel

    _, _, n_frames = mel.shape
    out = torch.zeros_like(mel)

    if shift_frames > 0:
        if shift_frames >= n_frames:
            return out
        out[:, :, shift_frames:] = mel[:, :, : n_frames - shift_frames]
        return out

    k = -shift_frames
    if k >= n_frames:
        return out
    out[:, :, : n_frames - k] = mel[:, :, k:]
    return out


def _mel_mask(mel: torch.Tensor, axis: int, mask_param: int) -> torch.Tensor:
    """SpecAugment: случайная маска по времени или частоте."""

    if mask_param <= 0:
        return mel

    out = mel.clone()
    _, n_mels, n_frames = out.shape

    if axis == 2:  # time
        if n_frames <= 1:
            return out
        width = random.randint(0, min(mask_param, n_frames - 1))
        if width == 0:
            return out
        start = random.randint(0, max(0, n_frames - width))
        out[:, :, start : start + width] = 0.0
        return out

    if axis == 1:  # freq
        if n_mels <= 1:
            return out
        width = random.randint(0, min(mask_param, n_mels - 1))
        if width == 0:
            return out
        start = random.randint(0, max(0, n_mels - width))
        out[:, start : start + width, :] = 0.0
        return out

    raise ValueError("axis должен быть 1 (freq) или 2 (time)")


class EmotionDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        augment: bool = False,
        augment_config: Optional[MelAugmentConfig] = None,
        waveform_noise_augment: bool = False,
        waveform_noise_config: Optional[WaveformNoiseAugmentConfig] = None,
        max_frames: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):

        self.df = dataframe.reset_index(drop=True)
        self.augment = bool(augment)
        self.augment_config = augment_config or MelAugmentConfig()
        self.waveform_noise_augment = bool(waveform_noise_augment)
        self.waveform_noise_config = waveform_noise_config or WaveformNoiseAugmentConfig()
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_cache = bool(use_cache and self.cache_dir is not None and not self.waveform_noise_augment)
        self.noise_manager: NoiseManager | None = None
        self.available_all_noise_types: list[str] = []
        self.available_real_noise_types: list[str] = []
        self.normalized_waveform_noise_types: tuple[str, ...] = ()

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.waveform_noise_augment:
            self._initialize_waveform_noise_augmentation()

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["path"]
        label = row["label"]

        if self.waveform_noise_augment:
            audio = self._load_preprocessed_audio(path)
            audio = self._apply_waveform_noise_augmentation(audio)
            mel = self._compute_mel_from_audio(audio)
        else:
            mel = self._load_or_compute_mel(path)

        if self.augment:
            mel = self._augment_mel(mel)
        if self.max_frames is not None and int(mel.shape[-1]) > int(self.max_frames):
            max_frames = int(self.max_frames)
            max_start = int(mel.shape[-1]) - max_frames
            start = random.randint(0, max_start) if self.augment else max_start // 2
            mel = mel[:, :, start : start + max_frames]

        label = torch.tensor(label, dtype=torch.long)

        return mel, label

    def _initialize_waveform_noise_augmentation(self) -> None:
        cfg = self.waveform_noise_config
        if not 0.0 <= float(cfg.noise_prob) <= 1.0:
            raise ValueError("waveform noise augmentation: noise_prob ?????? ???? ? ????????? [0, 1].")
        if float(cfg.snr_min) > float(cfg.snr_max):
            raise ValueError("waveform noise augmentation: snr_min ?? ?????? ???? ?????? snr_max.")

        normalized_noise_types = tuple(str(name).strip().lower() for name in cfg.noise_types if str(name).strip())
        if not normalized_noise_types:
            raise ValueError("waveform noise augmentation: ?????? noise_types ????.")

        self.noise_manager = NoiseManager(
            noise_dir=Path(cfg.noise_dir),
            sample_rate=int(cfg.sample_rate),
            random_seed=cfg.random_seed,
        )
        self.available_all_noise_types = self.noise_manager.list_available_noises()
        self.available_real_noise_types = [
            noise_name
            for noise_name in self.available_all_noise_types
            if noise_name not in self.noise_manager.SYNTHETIC_NOISES
        ]
        self.normalized_waveform_noise_types = normalized_noise_types

        available_variants = set(self.noise_manager.list_available_noise_variants())
        for noise_name in self.normalized_waveform_noise_types:
            if noise_name in {"white", "pink", "brown", "brownian", "real", "random"}:
                continue
            if noise_name in self.available_all_noise_types or noise_name in available_variants:
                continue
            raise ValueError(
                f"waveform noise augmentation: ??????????? ??? ???? {noise_name!r}. "
                f"????????? ????????: {', '.join(self.available_all_noise_types) or '???'}."
            )

        if "real" in self.normalized_waveform_noise_types and not self.available_real_noise_types:
            raise ValueError(
                f"waveform noise augmentation: ? ????? ????? ??? ???????? wav-??????: {cfg.noise_dir}"
            )

        if "random" in self.normalized_waveform_noise_types and not self.available_all_noise_types:
            raise ValueError("waveform noise augmentation: ?? ??????? ?? ?????? ?????????? ????.")

    def _cache_path(self, audio_path: str) -> Optional[Path]:
        if not self.use_cache or self.cache_dir is None:
            return None
        key = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.pt"

    def _load_or_compute_mel(self, audio_path: str) -> torch.Tensor:
        cache_path = self._cache_path(audio_path)
        if cache_path is not None and cache_path.exists():
            mel = torch.load(cache_path, map_location="cpu")
            if isinstance(mel, dict) and "mel" in mel:
                mel = mel["mel"]
            if not isinstance(mel, torch.Tensor):
                raise ValueError(f"???????????? ?????? ???? mel: {cache_path}")
            return mel

        audio = self._load_preprocessed_audio(audio_path)
        mel = self._compute_mel_from_audio(audio)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mel, cache_path)

        return mel

    def _load_preprocessed_audio(self, audio_path: str) -> np.ndarray:
        audio = load_audio(audio_path)
        audio = normalize_audio(audio)
        audio = trim_silence(audio)

        prepared_audio = np.asarray(audio, dtype=np.float32)
        if prepared_audio.size == 0:
            return np.zeros(160, dtype=np.float32)
        return prepared_audio

    def _compute_mel_from_audio(self, audio: np.ndarray) -> torch.Tensor:
        mel_np = extract_mel_spectrogram(np.asarray(audio, dtype=np.float32))
        mel = torch.from_numpy(mel_np).unsqueeze(0).float()
        return _standardize_mel(mel)

    def _apply_waveform_noise_augmentation(self, audio: np.ndarray) -> np.ndarray:
        if self.noise_manager is None:
            raise RuntimeError("waveform noise augmentation ?? ????????????????.")

        cfg = self.waveform_noise_config
        clean_audio = np.asarray(audio, dtype=np.float32)
        if random.random() >= float(cfg.noise_prob):
            return clean_audio

        max_attempts = 3
        duration = float(clean_audio.shape[0]) / float(self.noise_manager.sample_rate)

        for _ in range(max_attempts):
            try:
                selected_noise_name = self._sample_waveform_noise_name()
                snr_db = random.uniform(float(cfg.snr_min), float(cfg.snr_max))

                if selected_noise_name in self.noise_manager.SYNTHETIC_NOISES or selected_noise_name == "brownian":
                    synthetic_name = "brown" if selected_noise_name == "brownian" else selected_noise_name
                    noise = self.noise_manager.generate_synthetic_noise(synthetic_name, duration)
                else:
                    noise = self.noise_manager.get_real_noise(selected_noise_name, duration)

                noisy_audio = self.noise_manager.add_noise(clean_audio, noise, snr_db=snr_db)
                return normalize_audio(noisy_audio)
            except ValueError:
                continue

        return clean_audio

    def _sample_waveform_noise_name(self) -> str:
        if self.noise_manager is None:
            raise RuntimeError("waveform noise augmentation ?? ????????????????.")

        token = random.choice(self.normalized_waveform_noise_types)
        if token == "random":
            if not self.available_all_noise_types:
                raise ValueError("??? ?????? random ?? ??????? ????????? ?????.")
            return random.choice(self.available_all_noise_types)
        if token == "real":
            if not self.available_real_noise_types:
                raise ValueError("??? ?????? real ?? ??????? ?? ?????? ????????? ????.")
            return random.choice(self.available_real_noise_types)
        return token

    def _augment_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """??????????? ??? ???????? (noise/time-stretch/pitch-shift + SpecAugment)."""

        cfg = self.augment_config
        out = mel

        if random.random() < cfg.noise_prob:
            std = random.uniform(cfg.noise_std_min, cfg.noise_std_max)
            if std > 0:
                out = _mel_add_noise(out, std=std)

        if random.random() < cfg.time_stretch_prob:
            factor = random.uniform(cfg.time_stretch_min, cfg.time_stretch_max)
            out = _mel_time_stretch(out, factor=factor)

        if random.random() < cfg.pitch_shift_prob and cfg.pitch_shift_bins > 0:
            shift = random.randint(-cfg.pitch_shift_bins, cfg.pitch_shift_bins)
            out = _mel_pitch_shift(out, shift_bins=shift)

        if random.random() < cfg.time_shift_prob and float(cfg.time_shift_max_frac) > 0:
            max_shift = int(round(int(out.shape[-1]) * float(cfg.time_shift_max_frac)))
            if max_shift > 0:
                shift = random.randint(-max_shift, max_shift)
                out = _mel_time_shift(out, shift_frames=int(shift))

        if random.random() < cfg.time_mask_prob:
            out = _mel_mask(out, axis=2, mask_param=cfg.time_mask_param)
        if random.random() < cfg.freq_mask_prob:
            out = _mel_mask(out, axis=1, mask_param=cfg.freq_mask_param)

        return out


class MultimodalDataset(EmotionDataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        augment: bool = False,
        augment_config: Optional[MelAugmentConfig] = None,
        waveform_noise_augment: bool = False,
        waveform_noise_config: Optional[WaveformNoiseAugmentConfig] = None,
        max_frames: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        super().__init__(
            dataframe=dataframe,
            augment=augment,
            augment_config=augment_config,
            waveform_noise_augment=waveform_noise_augment,
            waveform_noise_config=waveform_noise_config,
            max_frames=max_frames,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["path"]
        label = row["label"]

        audio = self._load_preprocessed_audio(path)
        if self.waveform_noise_augment:
            audio = self._apply_waveform_noise_augmentation(audio)
            mel = self._compute_mel_from_audio(audio)
        else:
            mel = self._load_or_compute_mel(path)

        if self.augment:
            mel = self._augment_mel(mel)
        if self.max_frames is not None and int(mel.shape[-1]) > int(self.max_frames):
            max_frames = int(self.max_frames)
            max_start = int(mel.shape[-1]) - max_frames
            start = random.randint(0, max_start) if self.augment else max_start // 2
            mel = mel[:, :, start : start + max_frames]

        label = torch.tensor(label, dtype=torch.long)

        return audio, mel, label


# ===============================
# Выравнивает спектрограммы разной длины до одинакового размера.
# ===============================

def pad_mels_collate_fn(batch, pad_value=0.0):

    mels, labels = zip(*batch)

    max_frames = max(mel.shape[-1] for mel in mels)

    padded_mels = [
        F.pad(mel, (0, max_frames - mel.shape[-1]), value=pad_value)
        for mel in mels
    ]

    return torch.stack(padded_mels, dim=0), torch.stack(labels, dim=0)


# ===============================
# Версия collate, которая дополнительно возвращает длины (frames) до паддинга.
# Нужна для RNN/attention, чтобы корректно игнорировать padding.
# ===============================

def pad_mels_collate_fn_with_lengths(batch, pad_value=0.0):

    mels, labels = zip(*batch)

    lengths = torch.tensor([int(mel.shape[-1]) for mel in mels], dtype=torch.long)
    max_frames = int(lengths.max().item()) if len(lengths) > 0 else 0

    padded_mels = [
        F.pad(mel, (0, max_frames - mel.shape[-1]), value=pad_value)
        for mel in mels
    ]

    return torch.stack(padded_mels, dim=0), lengths, torch.stack(labels, dim=0)


# ===============================
# Выравнивает спектрограммы разной длины до одинакового размера.
# ===============================

def multimodal_collate_fn(batch, pad_value=0.0):

    audios, mels, labels = zip(*batch)

    max_frames = max(mel.shape[-1] for mel in mels)

    padded_mels = [
        F.pad(mel, (0, max_frames - mel.shape[-1]), value=pad_value)
        for mel in mels
    ]

    return list(audios), torch.stack(padded_mels, dim=0), torch.stack(labels, dim=0)


# ===============================
# Multimodal collate + lengths.
# ===============================

def multimodal_collate_fn_with_lengths(batch, pad_value=0.0):

    audios, mels, labels = zip(*batch)

    lengths = torch.tensor([int(mel.shape[-1]) for mel in mels], dtype=torch.long)
    max_frames = int(lengths.max().item()) if len(lengths) > 0 else 0

    padded_mels = [
        F.pad(mel, (0, max_frames - mel.shape[-1]), value=pad_value)
        for mel in mels
    ]

    return list(audios), torch.stack(padded_mels, dim=0), lengths, torch.stack(labels, dim=0)


# ===============================
#  Готовит три датасета для обычной модели (только спектрограммы)
# ===============================

def prepare_datasets(
    crema_path,
    ravdess_path,
    *,
    emotion_set: int = 8,
    augment: bool = False,
    augment_config: Optional[MelAugmentConfig] = None,
    waveform_noise_augment: bool = False,
    waveform_noise_config: Optional[WaveformNoiseAugmentConfig] = None,
    max_frames: Optional[int] = None,
    cache_dir: Optional[Path] = None,
):

    train_df, val_df, test_df, emotion_map = prepare_splits(
        crema_path,
        ravdess_path,
        verbose=True,
        emotion_set=emotion_set,
    )

    return (
        EmotionDataset(
            train_df,
            augment=augment,
            augment_config=augment_config,
            waveform_noise_augment=waveform_noise_augment,
            waveform_noise_config=waveform_noise_config,
            max_frames=max_frames,
            cache_dir=cache_dir,
        ),
        EmotionDataset(
            val_df,
            augment=False,
            augment_config=augment_config,
            waveform_noise_augment=False,
            waveform_noise_config=waveform_noise_config,
            max_frames=max_frames,
            cache_dir=cache_dir,
        ),
        EmotionDataset(
            test_df,
            augment=False,
            augment_config=augment_config,
            waveform_noise_augment=False,
            waveform_noise_config=waveform_noise_config,
            max_frames=max_frames,
            cache_dir=cache_dir,
        ),
        emotion_map,
    )

# ===============================
#  Готовит три датасета для мультимодальный модели (только спектрограммы)
# ===============================
def prepare_multimodal_datasets(
    crema_path,
    ravdess_path,
    *,
    emotion_map=None,
    emotion_set: int = 8,
    augment: bool = False,
    augment_config: Optional[MelAugmentConfig] = None,
    waveform_noise_augment: bool = False,
    waveform_noise_config: Optional[WaveformNoiseAugmentConfig] = None,
    max_frames: Optional[int] = None,
    cache_dir: Optional[Path] = None,
):

    train_df, val_df, test_df, emotion_map = prepare_splits(
        crema_path,
        ravdess_path,
        emotion_map=emotion_map,
        verbose=True,
        emotion_set=emotion_set,
    )

    return (
        MultimodalDataset(
            train_df,
            augment=augment,
            augment_config=augment_config,
            waveform_noise_augment=waveform_noise_augment,
            waveform_noise_config=waveform_noise_config,
            max_frames=max_frames,
            cache_dir=cache_dir,
        ),
        MultimodalDataset(
            val_df,
            augment=False,
            augment_config=augment_config,
            waveform_noise_augment=False,
            waveform_noise_config=waveform_noise_config,
            max_frames=max_frames,
            cache_dir=cache_dir,
        ),
        MultimodalDataset(
            test_df,
            augment=False,
            augment_config=augment_config,
            waveform_noise_augment=False,
            waveform_noise_config=waveform_noise_config,
            max_frames=max_frames,
            cache_dir=cache_dir,
        ),
        emotion_map,
    )

# ===============================
#  Создаёт "пачки" (batches) для подачи в нейросеть
# ===============================
def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=16,
    num_workers=0,
    *,
    balanced_sampling: bool = False,
    with_lengths: bool = False,
    pad_value: float = 0.0,
):

    collate = pad_mels_collate_fn_with_lengths if with_lengths else pad_mels_collate_fn
    sampler = None
    if balanced_sampling:
        # Балансируем классы через WeightedRandomSampler.
        # Это мягче, чем "жёсткие" class-weights в лоссе, и часто даёт лучший macro-F1.
        try:
            labels = train_dataset.df["label"].astype(int).tolist()
            if labels:
                counts = np.bincount(np.asarray(labels, dtype=np.int64))
                counts = np.maximum(counts, 1)
                sample_weights = [1.0 / float(counts[int(y)]) for y in labels]
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
        except Exception:
            sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda b: collate(b, pad_value=pad_value),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate(b, pad_value=pad_value),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate(b, pad_value=pad_value),
    )

    return train_loader, val_loader, test_loader

# ===============================
#  Создаёт "пачки" (batches) для подачи в нейросеть
# ===============================
def create_multimodal_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=4,
    num_workers=0,
    *,
    balanced_sampling: bool = False,
    with_lengths: bool = False,
    pad_value: float = 0.0,
):

    collate = multimodal_collate_fn_with_lengths if with_lengths else multimodal_collate_fn
    sampler = None
    if balanced_sampling:
        try:
            labels = train_dataset.df["label"].astype(int).tolist()
            if labels:
                counts = np.bincount(np.asarray(labels, dtype=np.int64))
                counts = np.maximum(counts, 1)
                sample_weights = [1.0 / float(counts[int(y)]) for y in labels]
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
        except Exception:
            sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda b: collate(b, pad_value=pad_value),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate(b, pad_value=pad_value),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate(b, pad_value=pad_value),
    )

    return train_loader, val_loader, test_loader
