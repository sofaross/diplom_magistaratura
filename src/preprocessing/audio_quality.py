from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.preprocessing.audio_processing import load_audio

DEFAULT_QUALITY_METRICS: tuple[str, ...] = (
    "duration_sec",
    "rms_db",
    "silence_ratio",
)


def _row_has_embedded_audio(row: pd.Series) -> bool:
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


def _load_audio_from_row(row: pd.Series, sample_rate: int) -> np.ndarray:
    if _row_has_embedded_audio(row):
        audio = np.asarray(row["hf_audio"], dtype=np.float32).reshape(-1)
        source_sample_rate = int(row.get("hf_sampling_rate", sample_rate))
        if source_sample_rate != int(sample_rate) and audio.size > 0:
            import librosa

            audio = librosa.resample(audio, orig_sr=int(source_sample_rate), target_sr=int(sample_rate))
        return np.asarray(audio, dtype=np.float32)

    audio_path = Path(str(row["path"]))
    return np.asarray(load_audio(audio_path, sample_rate=int(sample_rate)), dtype=np.float32)


def _infer_source_dataset(row: pd.Series) -> str:
    if isinstance(row, pd.Series):
        explicit_source = row.get("source_dataset")
        if (
            explicit_source is not None
            and not pd.isna(explicit_source)
            and str(explicit_source).strip()
            and str(explicit_source).strip().lower() != "nan"
        ):
            return str(explicit_source)

        path_value = str(row.get("path", ""))
    else:
        path_value = str(row)

    path_lower = path_value.lower()
    if "crema" in path_lower:
        return "CREMA-D"
    if "ravdess" in path_lower:
        return "RAVDESS"
    if path_lower.startswith("hf://aniemore__resd/") or "aniemore/resd" in path_lower:
        return "Aniemore/resd"
    return "unknown"


def _compute_iqr_bounds(values: pd.Series, iqr_multiplier: float) -> tuple[float, float, float, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = float(q3 - q1)
    lower = float(q1 - float(iqr_multiplier) * iqr)
    upper = float(q3 + float(iqr_multiplier) * iqr)
    return q1, q3, iqr, lower, upper


def compute_audio_quality_metrics(
    df: pd.DataFrame,
    *,
    sample_rate: int = 16000,
    silence_threshold: float = 0.01,
    clipping_threshold: float = 0.98,
) -> pd.DataFrame:
    """
    Считает простые метрики качества записи для каждого примера.

    Возвращает таблицу с колонками:
    - path
    - source_dataset
    - duration_sec
    - rms_db
    - silence_ratio
    - clipping_ratio
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("compute_audio_quality_metrics: ожидается pandas.DataFrame")
    if "path" not in df.columns:
        raise ValueError("compute_audio_quality_metrics: в датафрейме нет колонки 'path'")

    records: list[dict[str, float | str]] = []

    for _, row in df.iterrows():
        audio = _load_audio_from_row(row, sample_rate=int(sample_rate))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        duration_sec = float(audio.size) / float(sample_rate) if audio.size > 0 else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size > 0 else 0.0
        rms_db = float(20.0 * np.log10(max(rms, 1e-8)))

        max_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        if max_abs > 0.0:
            normalized_audio = audio / max_abs
            silence_ratio = float(np.mean(np.abs(normalized_audio) < float(silence_threshold)))
        else:
            silence_ratio = 1.0

        clipping_ratio = (
            float(np.mean(np.abs(audio) >= float(clipping_threshold))) if audio.size > 0 else 0.0
        )

        records.append(
            {
                "path": str(row["path"]),
                "source_dataset": _infer_source_dataset(row),
                "duration_sec": duration_sec,
                "rms_db": rms_db,
                "silence_ratio": silence_ratio,
                "clipping_ratio": clipping_ratio,
            }
        )

    return pd.DataFrame(records)


def build_quality_boxplot_bounds(
    metrics_df: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = DEFAULT_QUALITY_METRICS,
    iqr_multiplier: float = 1.5,
    group_column: str | None = None,
) -> pd.DataFrame:
    """
    Считает границы выбросов по методу IQR ("ящик с усами") для каждой метрики.
    """
    if not isinstance(metrics_df, pd.DataFrame):
        raise ValueError("build_quality_boxplot_bounds: ожидается pandas.DataFrame")

    rows: list[dict[str, float | str]] = []
    for metric_name in metric_columns:
        if metric_name not in metrics_df.columns:
            raise ValueError(f"build_quality_boxplot_bounds: нет колонки {metric_name!r}")

        if group_column is not None:
            if group_column not in metrics_df.columns:
                raise ValueError(f"build_quality_boxplot_bounds: нет колонки {group_column!r}")

            grouped_values = metrics_df.assign(
                **{group_column: metrics_df[group_column].fillna("unknown").astype(str)}
            ).groupby(group_column, dropna=False)
            for group_value, group_df in grouped_values:
                q1, q3, iqr, lower, upper = _compute_iqr_bounds(group_df[metric_name], iqr_multiplier=float(iqr_multiplier))
                rows.append(
                    {
                        "metric": metric_name,
                        group_column: str(group_value),
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "lower": lower,
                        "upper": upper,
                    }
                )
            continue

        q1, q3, iqr, lower, upper = _compute_iqr_bounds(metrics_df[metric_name], iqr_multiplier=float(iqr_multiplier))
        rows.append(
            {
                "metric": metric_name,
                "source_dataset": "all",
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": lower,
                "upper": upper,
            }
        )

    return pd.DataFrame(rows)


def filter_audio_quality_outliers(
    df: pd.DataFrame,
    *,
    metrics_df: pd.DataFrame | None = None,
    sample_rate: int = 16000,
    metric_columns: Sequence[str] = DEFAULT_QUALITY_METRICS,
    iqr_multiplier: float = 1.5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Удаляет аномальные записи по качеству на основе IQR/boxplot.

    Возвращает:
    - filtered_df: датафрейм без выбросов
    - metrics_df: рассчитанные метрики по всем исходным записям
    - bounds_df: таблица границ boxplot по каждой метрике
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("filter_audio_quality_outliers: ожидается pandas.DataFrame")
    if len(df) == 0:
        return df.copy(), pd.DataFrame(), pd.DataFrame()

    if metrics_df is None:
        metrics_df = compute_audio_quality_metrics(df, sample_rate=int(sample_rate))

    global_metric_columns = tuple(name for name in metric_columns if name != "duration_sec")
    bounds_parts: list[pd.DataFrame] = []

    if "duration_sec" in metric_columns and "source_dataset" in metrics_df.columns:
        bounds_parts.append(
            build_quality_boxplot_bounds(
                metrics_df,
                metric_columns=("duration_sec",),
                iqr_multiplier=float(iqr_multiplier),
                group_column="source_dataset",
            )
        )
    elif "duration_sec" in metric_columns:
        bounds_parts.append(
            build_quality_boxplot_bounds(
                metrics_df,
                metric_columns=("duration_sec",),
                iqr_multiplier=float(iqr_multiplier),
            )
        )

    if global_metric_columns:
        bounds_parts.append(
            build_quality_boxplot_bounds(
                metrics_df,
                metric_columns=global_metric_columns,
                iqr_multiplier=float(iqr_multiplier),
            )
        )

    bounds_df = (
        pd.concat(bounds_parts, ignore_index=True)
        if len(bounds_parts) > 0
        else pd.DataFrame(columns=["metric", "source_dataset", "q1", "q3", "iqr", "lower", "upper"])
    )

    analysis_df = metrics_df.copy()
    keep_mask = np.ones(len(metrics_df), dtype=bool)
    for metric_name in metric_columns:
        metric_values = pd.to_numeric(metrics_df[metric_name], errors="coerce").to_numpy(dtype=np.float64)
        metric_keep_mask = np.ones(len(metrics_df), dtype=bool)

        if metric_name == "duration_sec" and "source_dataset" in metrics_df.columns:
            source_values = metrics_df["source_dataset"].fillna("unknown").astype(str)
            metric_keep_mask &= np.isfinite(metric_values)
            for dataset_name in sorted(source_values.unique()):
                dataset_mask = source_values.to_numpy(dtype=str) == str(dataset_name)
                bound_row = bounds_df.loc[
                    (bounds_df["metric"] == metric_name) & (bounds_df["source_dataset"].astype(str) == str(dataset_name))
                ]
                if len(bound_row) != 1:
                    continue
                lower = float(bound_row.iloc[0]["lower"])
                upper = float(bound_row.iloc[0]["upper"])
                if not np.isfinite(lower) or not np.isfinite(upper):
                    continue
                metric_keep_mask[dataset_mask] &= (
                    (metric_values[dataset_mask] >= lower) & (metric_values[dataset_mask] <= upper)
                )
        else:
            bound_row = bounds_df.loc[
                (bounds_df["metric"] == metric_name) & (bounds_df["source_dataset"].astype(str) == "all")
            ]
            if len(bound_row) == 1:
                lower = float(bound_row.iloc[0]["lower"])
                upper = float(bound_row.iloc[0]["upper"])
                metric_keep_mask &= np.isfinite(metric_values)
                if np.isfinite(lower) and np.isfinite(upper):
                    metric_keep_mask &= metric_values >= lower
                    metric_keep_mask &= metric_values <= upper
            else:
                metric_keep_mask &= np.isfinite(metric_values)

        analysis_df[f"outlier_{metric_name}"] = ~metric_keep_mask
        keep_mask &= metric_keep_mask

    analysis_df["drop_by_quality"] = ~keep_mask
    dropped_paths = set(metrics_df.loc[~keep_mask, "path"].astype(str).tolist())
    filtered_df = df.loc[~df["path"].astype(str).isin(dropped_paths)].reset_index(drop=True)
    dropped_df = analysis_df.loc[analysis_df["drop_by_quality"]].copy()

    if verbose:
        for metric_name in metric_columns:
            flag_column = f"outlier_{metric_name}"
            if flag_column in analysis_df.columns:
                dropped_count = int(analysis_df[flag_column].sum())
                print(f"[качество] Отброшено по метрике {metric_name}: {dropped_count}")
        if "source_dataset" in dropped_df.columns and len(dropped_df) > 0:
            print("[качество] Отброшено по датасетам:")
            by_dataset = dropped_df["source_dataset"].value_counts()
            for dataset_name, count in by_dataset.items():
                print(f"  - {dataset_name}: {int(count)}")

    return filtered_df, metrics_df, bounds_df


__all__ = [
    "DEFAULT_QUALITY_METRICS",
    "build_quality_boxplot_bounds",
    "compute_audio_quality_metrics",
    "filter_audio_quality_outliers",
]
