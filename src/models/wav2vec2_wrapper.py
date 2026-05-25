from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor

from configs.model_runtime import DEFAULT_SPEECH_MODEL, LOCAL_FILES_ONLY


@dataclass(slots=True)
class Wav2Vec2Wrapper:
    """Лёгкая обёртка над загруженной Wav2Vec2 CTC-моделью и её processor."""

    model_name: str
    device: torch.device
    processor: Any
    model: Any
    sample_rate: int = 16000

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_SPEECH_MODEL,
        device: str | torch.device | None = None,
        sample_rate: int = 16000,
        local_files_only: bool = LOCAL_FILES_ONLY,
    ) -> "Wav2Vec2Wrapper":
        """Загружает processor и CTC-модель с учётом общего режима offline/online."""

        resolved_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        cls._raise_if_incomplete_cache_detected(model_name)

        try:
            processor = cls._load_processor(model_name, local_files_only=local_files_only)
            model = AutoModelForCTC.from_pretrained(
                model_name,
                local_files_only=bool(local_files_only),
            )
        except Exception as exc:
            hint = (
                "Ожидалась локальная папка модели или уже готовый локальный HuggingFace cache."
                if local_files_only
                else "Проверьте имя модели и доступность Hugging Face."
            )
            raise RuntimeError(
                f"Не удалось загрузить Wav2Vec2 CTC модель {model_name!r}. {hint}"
            ) from exc

        model.to(resolved_device)
        model.eval()

        return cls(
            model_name=str(model_name),
            device=resolved_device,
            processor=processor,
            model=model,
            sample_rate=int(sample_rate),
        )

    @staticmethod
    def _load_processor(model_name: str, *, local_files_only: bool) -> Any:
        try:
            return AutoProcessor.from_pretrained(
                model_name,
                local_files_only=bool(local_files_only),
            )
        except ImportError as exc:
            error_text = str(exc)
            if "pyctcdecode" not in error_text:
                raise
            return Wav2Vec2Processor.from_pretrained(
                model_name,
                local_files_only=bool(local_files_only),
            )

    @staticmethod
    def _raise_if_incomplete_cache_detected(model_name: str) -> None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_name.replace("/", "--"))
        if not cache_dir.exists():
            return

        incomplete_files = sorted(path for path in cache_dir.rglob("*.incomplete") if path.is_file())
        if not incomplete_files:
            return

        details = ", ".join(f"{path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)" for path in incomplete_files)
        raise RuntimeError(
            f"Для модели {model_name!r} найден недокачанный кэш Hugging Face: {details}. "
            f"Удалите каталог {cache_dir} и запустите загрузку заново."
        )

    @property
    def hidden_size(self) -> int:
        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        return int(self.base_model.config.hidden_size)

    @property
    def embedding_dim(self) -> int:
        return self.hidden_size

    @property
    def base_model(self) -> torch.nn.Module:
        if hasattr(self.model, "wav2vec2"):
            return self.model.wav2vec2
        for attr_name in ("hubert", "data2vec_audio"):
            if hasattr(self.model, attr_name):
                return getattr(self.model, attr_name)
        if isinstance(self.model, torch.nn.Module):
            return self.model
        raise RuntimeError("Не удалось определить базовую часть speech-модели.")


__all__ = ["Wav2Vec2Wrapper"]
