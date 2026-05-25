from __future__ import annotations

import importlib
import io
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

from configs.config import ProjectConfig
from src.constants.emotions import CREMA_EMOTIONS, RAVDESS_EMOTIONS, RESD_EMOTION_MAP_6
from src.preprocessing.audio_processing import load_audio
from src.preprocessing.audio_quality import filter_audio_quality_outliers


DEFAULT_DATASET_CONFIG = ProjectConfig()


# ===============================
# Загружаем CREMA-D датасет
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

        data.append({"path": str(file), "emotion": emotion})

    return pd.DataFrame(data, columns=["path", "emotion"])


# ===============================
# Загружаем RAVDESS датасет
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

        data.append({"path": str(file), "emotion": emotion})

    return pd.DataFrame(data, columns=["path", "emotion"])


# ===============================
# Нормализуем эмоции RESD
# ===============================
def _normalize_resd_emotion(emotion: str) -> str | None:
    normalized = str(emotion).strip().lower()
    if normalized == "enthusiasm":
        return None
    return RESD_EMOTION_MAP_6.get(normalized)


# ===============================
# Извлекаем аудио из HF payload
# ===============================
def _extract_hf_audio_payload(speech_payload) -> tuple[np.ndarray, int]:
    if isinstance(speech_payload, dict):
        if "array" in speech_payload:
            audio = np.asarray(speech_payload["array"], dtype=np.float32).reshape(-1)
            sample_rate = int(speech_payload.get("sampling_rate") or DEFAULT_DATASET_CONFIG.sample_rate)
            return audio, sample_rate
        if speech_payload.get("bytes") is not None:
            audio, sample_rate = sf.read(io.BytesIO(speech_payload["bytes"]), dtype="float32")
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            return audio, int(sample_rate)
        if speech_payload.get("path"):
            audio = load_audio(speech_payload["path"], sample_rate=int(DEFAULT_DATASET_CONFIG.sample_rate))
            return np.asarray(audio, dtype=np.float32).reshape(-1), int(DEFAULT_DATASET_CONFIG.sample_rate)
        raise ValueError("HF audio payload does not contain decoded audio data or a readable local path.")

    audio = np.asarray(speech_payload, dtype=np.float32).reshape(-1)
    return audio, int(DEFAULT_DATASET_CONFIG.sample_rate)


def _apply_quality_filter(
    df: pd.DataFrame,
    *,
    label: str,
    iqr_multiplier: float,
    verbose: bool,
) -> pd.DataFrame:
    before_filter_len = len(df)
    if verbose:
        print(f"{label} до фильтрации качества: {before_filter_len}")

    df, _, _ = filter_audio_quality_outliers(
        df,
        sample_rate=int(DEFAULT_DATASET_CONFIG.sample_rate),
        iqr_multiplier=float(iqr_multiplier),
        verbose=bool(verbose),
    )

    if verbose:
        print(f"{label} после фильтрации качества: {len(df)}")
        print(f"Отброшено по качеству: {before_filter_len - len(df)}")
    return df


# ===============================
# Загружаем RESD из Hugging Face
# ===============================
def load_resd_hf(
    dataset_name: str = "Aniemore/resd",
    *,
    split_names: tuple[str, ...] = ("train",),
    verbose: bool = True,
) -> pd.DataFrame:
    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as exc:
        raise ValueError(
            "Для использования Aniemore/resd установите пакет 'datasets': pip install datasets"
        ) from exc

    dataset_dict = datasets_module.load_dataset(str(dataset_name))
    records: list[dict[str, object]] = []
    dropped_enthusiasm = 0

    for split_name in split_names:
        if split_name not in dataset_dict:
            available = ", ".join(sorted(dataset_dict.keys()))
            raise ValueError(
                f"В датасете {dataset_name!r} нет split {split_name!r}. Доступные split: {available}."
            )

        split = dataset_dict[split_name]
        if hasattr(datasets_module, "Audio"):
            try:
                split = split.cast_column("speech", datasets_module.Audio(decode=False))
            except Exception:
                pass

        for index, row in enumerate(split):
            mapped_emotion = _normalize_resd_emotion(row.get("emotion", ""))
            if mapped_emotion is None:
                dropped_enthusiasm += 1
                continue

            audio, sample_rate = _extract_hf_audio_payload(row.get("speech"))
            sample_name = str(row.get("name") or row.get("path") or f"{split_name}_{index}")
            sample_id = f"hf://{str(dataset_name).replace('/', '__')}/{split_name}/{sample_name}"
            records.append(
                {
                    "path": sample_id,
                    "emotion": mapped_emotion,
                    "hf_audio": np.asarray(audio, dtype=np.float32),
                    "hf_sampling_rate": int(sample_rate),
                    "source_dataset": str(dataset_name),
                }
            )

    df = pd.DataFrame(records)
    if verbose:
        splits_text = ", ".join(split_names)
        print(f"RESD ({dataset_name}) примеров после drop_enthusiasm из split [{splits_text}]: {len(df)}")
        print(f"RESD отброшено emotion=enthusiasm: {dropped_enthusiasm}")
    return df


# ===============================
# Кодируем эмоции в label
# ===============================
def encode_labels(df, emotion_to_id=None):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("encode_labels: ожидается pandas.DataFrame")
    if "emotion" not in df.columns:
        raise ValueError("encode_labels: в датафрейме нет колонки 'emotion'")
    if len(df) == 0:
        raise ValueError("encode_labels: датасет пустой (0 примеров).")

    if emotion_to_id is None:
        emotions = sorted(df["emotion"].unique())
        emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotions)}

    df = df.copy()
    df["label"] = df["emotion"].map(emotion_to_id)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df, emotion_to_id


# ===============================
# Делим датасет на train/val/test
# ===============================
def split_dataset(df):
    if len(df) == 0:
        raise ValueError("split_dataset: датасет пустой (0 примеров)")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["label"],
        random_state=42,
    )

    return train_df, val_df, test_df


# ===============================
# Собираем итоговые таблицы
# ===============================
def prepare_splits(
    crema_path,
    ravdess_path,
    emotion_map=None,
    verbose=True,
    emotion_set: int = 6,
    use_resd: bool = False,
    resd_mode: str = "full_mix",
    resd_dataset_name: str = "Aniemore/resd",
    resd_splits: tuple[str, ...] = ("train",),
    quality_filter: bool = True,
    quality_filter_iqr_multiplier: float = 1.5,
):
    crema_df = load_crema_d(crema_path)
    ravdess_df = load_ravdess(ravdess_path)

    if verbose:
        print(f"CREMA-D примеров: {len(crema_df)}")
        print(f"RAVDESS примеров: {len(ravdess_df)}")

    df = pd.concat([crema_df, ravdess_df], ignore_index=True)

    if emotion_set == 6:
        allowed = set(CREMA_EMOTIONS.values())
        df = df[df["emotion"].isin(allowed)].reset_index(drop=True)
    elif emotion_set != 8:
        raise ValueError("emotion_set должен быть 6 или 8")

    resd_mode = str(resd_mode).strip().lower()
    if resd_mode not in {"full_mix", "train_only"}:
        raise ValueError("resd_mode должен быть 'full_mix' или 'train_only'.")

    resd_df = None
    if bool(use_resd):
        if int(emotion_set) != 6:
            raise ValueError("Aniemore/resd сейчас поддерживается только для emotion_set=6.")
        resd_df = load_resd_hf(
            dataset_name=str(resd_dataset_name),
            split_names=tuple(str(name) for name in resd_splits),
            verbose=bool(verbose),
        )
        if verbose:
            print(f"RESD примеров: {len(resd_df)}")
        if resd_mode == "full_mix":
            df = pd.concat([df, resd_df], ignore_index=True)

    if verbose:
        print("Всего примеров:", len(df))

    if bool(quality_filter):
        filter_label = "Общий набор данных" if not (bool(use_resd) and resd_mode == "train_only") else "Базовый набор данных"
        df = _apply_quality_filter(
            df,
            label=filter_label,
            iqr_multiplier=float(quality_filter_iqr_multiplier),
            verbose=bool(verbose),
        )

    df, emotion_map = encode_labels(df, emotion_to_id=emotion_map)
    train_df, val_df, test_df = split_dataset(df)

    if bool(use_resd) and resd_mode == "train_only" and resd_df is not None and len(resd_df) > 0:
        if bool(quality_filter):
            resd_df = _apply_quality_filter(
                resd_df,
                label="RESD train-only",
                iqr_multiplier=float(quality_filter_iqr_multiplier),
                verbose=bool(verbose),
            )
        if len(resd_df) > 0:
            resd_df, _ = encode_labels(resd_df, emotion_to_id=emotion_map)
            train_df = pd.concat([train_df, resd_df], ignore_index=True)
            if verbose:
                print(f"RESD добавлено только в обучение: {len(resd_df)}")

    if verbose:
        print("Размер обучения:", len(train_df))
        print("Размер проверки:", len(val_df))
        print("Размер экзамена:", len(test_df))

    return train_df, val_df, test_df, emotion_map


# ===============================
# Facade для табличной подготовки
# ===============================
class DatasetTableBuilder:
    load_crema_d = staticmethod(load_crema_d)
    load_ravdess = staticmethod(load_ravdess)
    load_resd_hf = staticmethod(load_resd_hf)
    encode_labels = staticmethod(encode_labels)
    split_dataset = staticmethod(split_dataset)
    prepare_splits = staticmethod(prepare_splits)


__all__ = [
    "DEFAULT_DATASET_CONFIG",
    "RESD_EMOTION_MAP_6",
    "DatasetTableBuilder",
    "load_crema_d",
    "load_ravdess",
    "load_resd_hf",
    "encode_labels",
    "split_dataset",
    "prepare_splits",
]
