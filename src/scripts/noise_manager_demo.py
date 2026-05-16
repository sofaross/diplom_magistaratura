from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import ProjectConfig
from src.noise.noise_manager import NoiseManager

DEFAULT_CONFIG = ProjectConfig()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _parse_csv(values: str) -> list[str]:
    return [item.strip() for item in str(values).split(",") if item.strip()]


def _parse_snr_csv(values: str) -> list[float]:
    return [float(item.strip()) for item in str(values).split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.scripts.noise_manager_demo")
    parser.add_argument("--noise-dir", default=str(DEFAULT_CONFIG.noise_dir), help="Папка с реальными шумами.")
    parser.add_argument("--clean-dir", default=str(DEFAULT_CONFIG.clean_recordings_dir), help="Папка с чистыми записями.")
    parser.add_argument("--output-root", default=str(DEFAULT_CONFIG.noisy_recordings_dir), help="Куда сохранять зашумлённые записи.")
    parser.add_argument(
        "--noise-types",
        default="white,pink,brown",
        help="Типы шумов через запятую, например: white,pink,metro",
    )
    parser.add_argument(
        "--snr-levels",
        default="20,10,0",
        help="Уровни SNR через запятую, например: 20,15,10,5,0",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Частота дискретизации.")
    parser.add_argument("--seed", type=int, default=42, help="Зерно генератора случайных чисел.")
    args = parser.parse_args(argv)

    noise_dir = _resolve_repo_path(args.noise_dir)
    clean_dir = _resolve_repo_path(args.clean_dir)
    output_root = _resolve_repo_path(args.output_root)
    noise_types = _parse_csv(args.noise_types)
    snr_levels = _parse_snr_csv(args.snr_levels)

    manager = NoiseManager(
        noise_dir=noise_dir,
        sample_rate=int(args.sample_rate),
        random_seed=int(args.seed),
    )

    print("[noise-demo] Доступные шумы:", ", ".join(manager.list_available_noises()))

    try:
        manager.create_noisy_dataset(
            clean_dir=clean_dir,
            output_root=output_root,
            noise_types=noise_types,
            snr_levels=snr_levels,
        )
    except Exception as exc:
        print(f"[noise-demo] Ошибка: {exc}")
        return 1

    print(f"[noise-demo] Готово. Результат сохранён в: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

