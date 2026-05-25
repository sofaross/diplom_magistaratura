from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from configs.config import ProjectConfig
from src.features.feature_extraction import extract_mel_spectrogram
from src.noise.noise_manager import NoiseManager
from src.preprocessing.audio_processing import load_audio, normalize_audio, trim_silence
from src.utils.dataset_sources import DEFAULT_DATASET_CONFIG, prepare_splits


# ===============================
# Конфиг mel-аугментаций
# ===============================
@dataclass(frozen=True)
class MelAugmentConfig:
    noise_prob: float = 0.5
    noise_std_min: float = 0.0
    noise_std_max: float = 0.25
    time_stretch_prob: float = 0.35
    time_stretch_min: float = 0.85
    time_stretch_max: float = 1.20
    pitch_shift_prob: float = 0.35
    pitch_shift_bins: int = 6
    time_mask_prob: float = 0.35
    time_mask_param: int = 30
    freq_mask_prob: float = 0.35
    freq_mask_param: int = 12
    time_shift_prob: float = 0.25
    time_shift_max_frac: float = 0.10


# ===============================
# Конфиг waveform noise augmentation
# ===============================
@dataclass(frozen=True)
class WaveformNoiseAugmentConfig:
    noise_prob: float = 0.5
    noise_dir: Path = DEFAULT_DATASET_CONFIG.noise_dir
    noise_types: tuple[str, ...] = ("white", "pink", "brown", "real")
    snr_min: float = 5.0
    snr_max: float = 20.0
    sample_rate: int = DEFAULT_DATASET_CONFIG.sample_rate
    random_seed: int | None = None


# ===============================
# Вспомогательные функции для mel
# ===============================
def _standardize_mel(mel: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = mel.mean()
    std = mel.std(unbiased=False)
    if not torch.isfinite(std) or float(std) < eps:
        return mel - mean
    return (mel - mean) / (std + eps)


def _mel_add_noise(mel: torch.Tensor, std: float) -> torch.Tensor:
    return mel + torch.randn_like(mel) * float(std)


def _mel_time_stretch(mel: torch.Tensor, factor: float) -> torch.Tensor:
    factor = float(factor)
    if factor <= 0:
        return mel

    _, n_mels, n_frames = mel.shape
    new_frames = max(1, int(round(n_frames * factor)))
    if new_frames == n_frames:
        return mel

    x = mel.unsqueeze(0)
    x = F.interpolate(x, size=(n_mels, new_frames), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def _mel_pitch_shift(mel: torch.Tensor, shift_bins: int) -> torch.Tensor:
    if shift_bins == 0:
        return mel

    _, n_mels, _ = mel.shape
    out = torch.zeros_like(mel)

    if shift_bins > 0:
        if shift_bins >= n_mels:
            return out
        out[:, shift_bins:, :] = mel[:, : n_mels - shift_bins, :]
    else:
        k = -shift_bins
        if k >= n_mels:
            return out
        out[:, : n_mels - k, :] = mel[:, k:, :]

    return out


def _mel_time_shift(mel: torch.Tensor, shift_frames: int) -> torch.Tensor:
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
    if mask_param <= 0:
        return mel

    out = mel.clone()
    _, n_mels, n_frames = out.shape

    if axis == 2:
        if n_frames <= 1:
            return out
        width = random.randint(0, min(mask_param, n_frames - 1))
        if width == 0:
            return out
        start = random.randint(0, max(0, n_frames - width))
        out[:, :, start : start + width] = 0.0
        return out

    if axis == 1:
        if n_mels <= 1:
            return out
        width = random.randint(0, min(mask_param, n_mels - 1))
        if width == 0:
            return out
        start = random.randint(0, max(0, n_mels - width))
        out[:, start : start + width, :] = 0.0
        return out

    raise ValueError("axis должен быть 1 (freq) или 2 (time)")


# ===============================
# Базовый PyTorch датасет эмоций
# ===============================
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
        label = row["label"]

        if self.waveform_noise_augment:
            audio = self._load_preprocessed_audio(row)
            audio = self._apply_waveform_noise_augmentation(audio)
            mel = self._compute_mel_from_audio(audio)
        else:
            mel = self._load_or_compute_mel(row)

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
            raise ValueError("waveform noise augmentation: noise_prob должен быть в диапазоне [0, 1].")
        if float(cfg.snr_min) > float(cfg.snr_max):
            raise ValueError("waveform noise augmentation: snr_min не должен быть больше snr_max.")

        normalized_noise_types = tuple(str(name).strip().lower() for name in cfg.noise_types if str(name).strip())
        if not normalized_noise_types:
            raise ValueError("waveform noise augmentation: список noise_types пуст.")

        self.noise_manager = NoiseManager(
            noise_dir=Path(cfg.noise_dir),
            sample_rate=int(cfg.sample_rate),
            random_seed=cfg.random_seed,
        )
        self.available_all_noise_types = self.noise_manager.list_available_noises()
        self.available_real_noise_types = [
            noise_name for noise_name in self.available_all_noise_types if noise_name not in self.noise_manager.SYNTHETIC_NOISES
        ]
        self.normalized_waveform_noise_types = normalized_noise_types

        available_variants = set(self.noise_manager.list_available_noise_variants())
        for noise_name in self.normalized_waveform_noise_types:
            if noise_name in {"white", "pink", "brown", "brownian", "real", "random"}:
                continue
            if noise_name in self.available_all_noise_types or noise_name in available_variants:
                continue
            raise ValueError(
                f"waveform noise augmentation: неизвестный тип шума {noise_name!r}. "
                f"Доступные варианты: {', '.join(self.available_all_noise_types) or 'none'}."
            )

        if "real" in self.normalized_waveform_noise_types and not self.available_real_noise_types:
            raise ValueError(f"waveform noise augmentation: в папке нет реальных wav-шумов: {cfg.noise_dir}")

        if "random" in self.normalized_waveform_noise_types and not self.available_all_noise_types:
            raise ValueError("waveform noise augmentation: нет доступных типов шума.")

    def _cache_path(self, row) -> Optional[Path]:
        if not self.use_cache or self.cache_dir is None:
            return None
        sample_id = self._resolve_sample_id(row)
        key = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.pt"

    def _load_or_compute_mel(self, row) -> torch.Tensor:
        cache_path = self._cache_path(row)
        if cache_path is not None and cache_path.exists():
            mel = torch.load(cache_path, map_location="cpu")
            if isinstance(mel, dict) and "mel" in mel:
                mel = mel["mel"]
            if not isinstance(mel, torch.Tensor):
                raise ValueError(f"Некорректный формат кэша mel: {cache_path}")
            return mel

        audio = self._load_preprocessed_audio(row)
        mel = self._compute_mel_from_audio(audio)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mel, cache_path)

        return mel

    def _load_preprocessed_audio(self, row) -> np.ndarray:
        if self._row_has_embedded_audio(row):
            audio = np.asarray(row["hf_audio"], dtype=np.float32).reshape(-1)
            source_sample_rate = int(row.get("hf_sampling_rate", DEFAULT_DATASET_CONFIG.sample_rate))
            if source_sample_rate != int(DEFAULT_DATASET_CONFIG.sample_rate):
                audio = librosa.resample(
                    audio,
                    orig_sr=int(source_sample_rate),
                    target_sr=int(DEFAULT_DATASET_CONFIG.sample_rate),
                )
        else:
            audio_path = self._resolve_sample_id(row)
            audio = load_audio(audio_path)

        audio = normalize_audio(audio)
        audio = trim_silence(audio)

        prepared_audio = np.asarray(audio, dtype=np.float32)
        if prepared_audio.size == 0:
            return np.zeros(160, dtype=np.float32)
        return prepared_audio

    @staticmethod
    def _row_has_embedded_audio(row) -> bool:
        if not isinstance(row, pd.Series) or "hf_audio" not in row.index:
            return False

        value = row["hf_audio"]
        if value is None:
            return False
        if isinstance(value, float) and pd.isna(value):
            return False
        if isinstance(value, np.ndarray):
            return value.size > 0
        if isinstance(value, (list, tuple, bytes, bytearray)):
            return len(value) > 0
        return not pd.isna(value)

    @staticmethod
    def _resolve_sample_id(row) -> str:
        if isinstance(row, pd.Series):
            return str(row["path"])
        return str(row)

    def _compute_mel_from_audio(self, audio: np.ndarray) -> torch.Tensor:
        mel_np = extract_mel_spectrogram(np.asarray(audio, dtype=np.float32))
        mel = torch.from_numpy(mel_np).unsqueeze(0).float()
        return _standardize_mel(mel)

    def _apply_waveform_noise_augmentation(self, audio: np.ndarray) -> np.ndarray:
        if self.noise_manager is None:
            raise RuntimeError("waveform noise augmentation is not initialized.")

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
            raise RuntimeError("waveform noise augmentation is not initialized.")

        token = random.choice(self.normalized_waveform_noise_types)
        if token == "random":
            if not self.available_all_noise_types:
                raise ValueError("For random noise there are no available noise types.")
            return random.choice(self.available_all_noise_types)
        if token == "real":
            if not self.available_real_noise_types:
                raise ValueError("For real noise there are no available real-noise variants.")
            return random.choice(self.available_real_noise_types)
        return token

    def _augment_mel(self, mel: torch.Tensor) -> torch.Tensor:
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


# ===============================
# Мультимодальный датасет
# ===============================
class MultimodalDataset(EmotionDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row["label"]

        audio = self._load_preprocessed_audio(row)
        if self.waveform_noise_augment:
            audio = self._apply_waveform_noise_augmentation(audio)
            mel = self._compute_mel_from_audio(audio)
        else:
            mel = self._load_or_compute_mel(row)

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
# Collate-функции для батчей
# ===============================
def pad_mels_collate_fn(batch, pad_value=0.0):
    mels, labels = zip(*batch)
    max_frames = max(mel.shape[-1] for mel in mels)
    padded_mels = [F.pad(mel, (0, max_frames - mel.shape[-1]), value=pad_value) for mel in mels]
    return torch.stack(padded_mels, dim=0), torch.stack(labels, dim=0)


def pad_mels_collate_fn_with_lengths(batch, pad_value=0.0):
    mels, labels = zip(*batch)
    lengths = torch.tensor([int(mel.shape[-1]) for mel in mels], dtype=torch.long)
    max_frames = int(lengths.max().item()) if len(lengths) > 0 else 0
    padded_mels = [F.pad(mel, (0, max_frames - mel.shape[-1]), value=pad_value) for mel in mels]
    return torch.stack(padded_mels, dim=0), lengths, torch.stack(labels, dim=0)


def multimodal_collate_fn(batch, pad_value=0.0):
    audios, mels, labels = zip(*batch)
    max_frames = max(mel.shape[-1] for mel in mels)
    padded_mels = [F.pad(mel, (0, max_frames - mel.shape[-1]), value=pad_value) for mel in mels]
    return list(audios), torch.stack(padded_mels, dim=0), torch.stack(labels, dim=0)


def multimodal_collate_fn_with_lengths(batch, pad_value=0.0):
    audios, mels, labels = zip(*batch)
    lengths = torch.tensor([int(mel.shape[-1]) for mel in mels], dtype=torch.long)
    max_frames = int(lengths.max().item()) if len(lengths) > 0 else 0
    padded_mels = [F.pad(mel, (0, max_frames - mel.shape[-1]), value=pad_value) for mel in mels]
    return list(audios), torch.stack(padded_mels, dim=0), lengths, torch.stack(labels, dim=0)


# ===============================
# Создаем emotion datasets
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
    use_resd: bool = False,
    resd_mode: str = "full_mix",
    resd_dataset_name: str = "Aniemore/resd",
    resd_splits: tuple[str, ...] = ("train",),
    quality_filter: bool = True,
    quality_filter_iqr_multiplier: float = 1.5,
):
    train_df, val_df, test_df, emotion_map = prepare_splits(
        crema_path,
        ravdess_path,
        verbose=True,
        emotion_set=emotion_set,
        use_resd=bool(use_resd),
        resd_mode=str(resd_mode),
        resd_dataset_name=str(resd_dataset_name),
        resd_splits=tuple(str(name) for name in resd_splits),
        quality_filter=bool(quality_filter),
        quality_filter_iqr_multiplier=float(quality_filter_iqr_multiplier),
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
# Создаем multimodal datasets
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
    use_resd: bool = False,
    resd_mode: str = "full_mix",
    resd_dataset_name: str = "Aniemore/resd",
    resd_splits: tuple[str, ...] = ("train",),
    quality_filter: bool = True,
    quality_filter_iqr_multiplier: float = 1.5,
):
    train_df, val_df, test_df, emotion_map = prepare_splits(
        crema_path,
        ravdess_path,
        emotion_map=emotion_map,
        verbose=True,
        emotion_set=emotion_set,
        use_resd=bool(use_resd),
        resd_mode=str(resd_mode),
        resd_dataset_name=str(resd_dataset_name),
        resd_splits=tuple(str(name) for name in resd_splits),
        quality_filter=bool(quality_filter),
        quality_filter_iqr_multiplier=float(quality_filter_iqr_multiplier),
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
# Создаем DataLoader для emotion model
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
# Создаем DataLoader для multimodal model
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


# ===============================
# Facade для runtime-части датасетов
# ===============================
class DatasetFactory:
    prepare_datasets = staticmethod(prepare_datasets)
    prepare_multimodal_datasets = staticmethod(prepare_multimodal_datasets)
    create_dataloaders = staticmethod(create_dataloaders)
    create_multimodal_dataloaders = staticmethod(create_multimodal_dataloaders)


__all__ = [
    "MelAugmentConfig",
    "WaveformNoiseAugmentConfig",
    "EmotionDataset",
    "MultimodalDataset",
    "DatasetFactory",
    "prepare_datasets",
    "prepare_multimodal_datasets",
    "create_dataloaders",
    "create_multimodal_dataloaders",
    "pad_mels_collate_fn",
    "pad_mels_collate_fn_with_lengths",
    "multimodal_collate_fn",
    "multimodal_collate_fn_with_lengths",
]
