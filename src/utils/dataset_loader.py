from pathlib import Path
import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.preprocessing.audio_preprocessing import load_audio, normalize_audio, trim_silence
from src.features.feature_extraction import extract_mel_spectrogram


# ===============================
# Эмоции
# ===============================

CREMA_EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}


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
            "dataset": "crema_d"
        })

    df = pd.DataFrame(data, columns=["path", "emotion", "dataset"])

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
            "dataset": "ravdess"
        })

    df = pd.DataFrame(data, columns=["path", "emotion", "dataset"])

    return df


# ===============================
# Объединение датасетов
# ===============================

def load_emotion_datasets(crema_path, ravdess_path):

    crema_df = load_crema_d(crema_path)

    ravdess_df = load_ravdess(ravdess_path)

    dataset = pd.concat([crema_df, ravdess_df], ignore_index=True)

    return dataset


# ===============================
# Кодирование эмоций
# ===============================

def encode_labels(df, emotion_to_id=None):

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

def prepare_splits(crema_path, ravdess_path, emotion_map=None, verbose=True):

    if verbose:
        print("Загружаем наборы данных...")

    df = load_emotion_datasets(crema_path, ravdess_path)

    if verbose:
        print("Всего примеров:", len(df))

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

class EmotionDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["path"]
        label = row["label"]

        audio = load_audio(path)

        audio = normalize_audio(audio)
        audio = trim_silence(audio)

        mel = extract_mel_spectrogram(audio)

        mel = torch.from_numpy(mel).unsqueeze(0).float()

        label = torch.tensor(label, dtype=torch.long)

        return mel, label


# ===============================
# Мультимодальный набор данных (аудио + спектрограмма)
# Улучшенная версия датасета, которая возвращает два представления звука:
#     1. Сырой аудиосигнал (для обработки звуковыми моделями)
#     2. Спектрограмму (для обработки как изображение)
# ===============================

class MultimodalDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["path"]
        label = row["label"]

        audio = load_audio(path)
        audio = normalize_audio(audio)
        audio = trim_silence(audio)

        mel = extract_mel_spectrogram(audio)
        mel = torch.from_numpy(mel).unsqueeze(0).float()

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
#  Готовит три датасета для обычной модели (только спектрограммы)
# ===============================

def prepare_datasets(crema_path, ravdess_path):

    train_df, val_df, test_df, emotion_map = prepare_splits(
        crema_path,
        ravdess_path,
        verbose=True,
    )

    return (
        EmotionDataset(train_df),
        EmotionDataset(val_df),
        EmotionDataset(test_df),
        emotion_map,
    )

# ===============================
#  Готовит три датасета для мультимодальный модели (только спектрограммы)
# ===============================
def prepare_multimodal_datasets(crema_path, ravdess_path):

    train_df, val_df, test_df, emotion_map = prepare_splits(
        crema_path,
        ravdess_path,
        verbose=True,
    )

    return (
        MultimodalDataset(train_df),
        MultimodalDataset(val_df),
        MultimodalDataset(test_df),
        emotion_map,
    )

# ===============================
#  Создаёт "пачки" (batches) для подачи в нейросеть
# ===============================
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16, num_workers=0):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_mels_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_mels_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_mels_collate_fn,
    )

    return train_loader, val_loader, test_loader

# ===============================
#  Создаёт "пачки" (batches) для подачи в нейросеть
# ===============================
def create_multimodal_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=4, num_workers=0):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
    )

    return train_loader, val_loader, test_loader
