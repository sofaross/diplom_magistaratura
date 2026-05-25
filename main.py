from __future__ import annotations

import argparse
import json
from pathlib import Path


def _smoke_check() -> None:
    import numpy as np
    import torch

    from src.models.emotion_model import EmotionModel
    from src.preprocessing.audio_processing import normalize_audio
    from src.utils.dataset_loader import pad_mels_collate_fn

    zeros = np.zeros(10, dtype=np.float32)
    normalized = normalize_audio(zeros)
    assert np.isfinite(normalized).all(), "normalize_audio() produced non-finite values"

    model = EmotionModel(num_emotions=6)
    x = torch.randn(2, 1, 128, 200)
    y = model(x)
    assert y.shape == (2, 6), f"Unexpected EmotionModel output shape: {tuple(y.shape)}"

    a = torch.zeros(1, 128, 10)
    b = torch.zeros(1, 128, 15)
    xb, yb = pad_mels_collate_fn([(a, torch.tensor(1)), (b, torch.tensor(2))])
    assert xb.shape == (2, 1, 128, 15), f"Unexpected collated mel shape: {tuple(xb.shape)}"
    assert yb.tolist() == [1, 2], f"Unexpected collated labels: {yb.tolist()}"

    print("src smoke-check: OK")


def _print_runtime_mode(config, *, speech_model_override: str | None = None) -> None:
    mode = "OFFLINE" if bool(config.local_files_only) else "ONLINE"
    print(f"[main] Режим запуска моделей распознавания речи: {mode}")

def _run_pipeline(args: argparse.Namespace) -> int:
    from configs.config import ProjectConfig
    from src.pipeline import MultimodalPipeline

    config = ProjectConfig()
    _print_runtime_mode(config, speech_model_override=args.speech_model)
    pipeline = MultimodalPipeline(
        config=config,
        speech_model_name=args.speech_model,
        emotion_model_path=args.emotion_model,
        emotion_map_path=args.emotion_map,
    )

    if args.list_noises:
        print("\n".join(pipeline.list_available_noises()))
        return 0

    if args.list_noise_variants:
        print("\n".join(pipeline.list_available_noise_variants()))
        return 0

    if not args.audio:
        raise SystemExit("You must provide --audio or use --list-noises / --list-noise-variants / --smoke.")
    if args.noise and args.noise_file:
        raise SystemExit("Use either --noise or --noise-file, not both.")

    result = pipeline.process_audio_file(
        args.audio,
        noise=args.noise,
        snr_db=args.snr,
        noise_file=args.noise_file,
        speech_language=args.speech_language,
    )

    rendered = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    print(rendered)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    return 0 if not result.errors else 1


def _run_interactive(args: argparse.Namespace) -> int:
    from configs.config import ProjectConfig
    from src.scripts.run_interactive_pipeline import InteractivePipelineRunner

    config = ProjectConfig()
    _print_runtime_mode(config, speech_model_override=args.speech_model)
    runner = InteractivePipelineRunner(
        speech_model_name=args.speech_model,
        emotion_model_path=args.emotion_model,
        emotion_map_path=args.emotion_map,
        default_speech_language=args.speech_language,
    )
    return runner.run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python main.py")
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke-check.")
    parser.add_argument("--interactive", action="store_true", help="Run the interactive console scenario.")
    parser.add_argument("--audio", default=None, help="Path to input audio file.")
    parser.add_argument("--noise", default=None, help="Noise type or 'random'.")
    parser.add_argument("--noise-file", default=None, help="Path to an external noise audio file.")
    parser.add_argument("--snr", type=float, default=10.0, help="Target SNR in dB for noise injection.")
    parser.add_argument("--speech-language", default="en", help="ASR language: ru or en.")
    parser.add_argument("--speech-model", default=None, help="Optional explicit HuggingFace ASR model override.")
    parser.add_argument("--emotion-model", default=None, help="Optional path to emotion checkpoint.")
    parser.add_argument("--emotion-map", default=None, help="Optional path to emotion label map.")
    parser.add_argument("--output-json", default=None, help="Optional path to save JSON result.")
    parser.add_argument("--list-noises", action="store_true", help="Print available noise categories and exit.")
    parser.add_argument(
        "--list-noise-variants",
        action="store_true",
        help="Print exact real-noise variants and exit.",
    )
    args = parser.parse_args()

    if args.smoke:
        _smoke_check()
        return

    if args.interactive:
        if args.audio is not None or args.noise is not None or args.noise_file is not None:
            raise SystemExit("Interactive mode does not use --audio, --noise, or --noise-file.")
        raise SystemExit(_run_interactive(args))

    if (
        args.audio is not None
        or args.list_noises
        or args.list_noise_variants
        or args.noise is not None
        or args.noise_file is not None
    ):
        raise SystemExit(_run_pipeline(args))

    parser.print_help()


if __name__ == "__main__":
    main()
