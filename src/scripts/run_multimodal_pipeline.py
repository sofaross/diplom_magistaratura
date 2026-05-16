from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import ProjectConfig
from src.services.audio_input_service import AudioInputService
from src.services.emotion_recognition_service import EmotionRecognitionService
from src.services.multimodal_classification_service import MultimodalClassificationService
from src.services.noise_service import NoiseService
from src.services.speech_recognition_service import SpeechRecognitionService


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    config = ProjectConfig()

    parser = argparse.ArgumentParser(prog="python -m src.scripts.run_multimodal_pipeline")
    parser.add_argument("--audio-file", default=None, help="Path to source audio file.")
    parser.add_argument("--microphone-duration", type=float, default=None, help="Record audio from microphone.")
    parser.add_argument("--noise-mode", choices=["none", "random", "selected"], default="none")
    parser.add_argument("--noise-type", default=None, help="Noise category or exact variant for selected mode.")
    parser.add_argument("--snr-db", type=float, default=10.0, help="Target SNR in dB.")
    parser.add_argument("--speech-language", default=str(config.default_speech_language), help="ASR language: ru or en.")
    parser.add_argument("--list-noises", action="store_true", help="Print available noise categories and exit.")
    parser.add_argument(
        "--list-noise-variants",
        action="store_true",
        help="Print exact real-noise file variants and exit.",
    )
    parser.add_argument("--speech-model", default=None, help="Optional explicit HuggingFace ASR model override.")
    parser.add_argument("--emotion-model", default=str(config.emotion_checkpoint_path))
    parser.add_argument("--emotion-map", default=str(config.emotion_map_path))
    parser.add_argument("--output-json", default=None, help="Save result to JSON file.")
    args = parser.parse_args(argv)

    audio_input_service = AudioInputService(config=config)
    noise_service = NoiseService(config=config)
    speech_service = SpeechRecognitionService(model_name=args.speech_model, preprocess=False, config=config)
    emotion_service = EmotionRecognitionService(
        emotion_model_path=_resolve_repo_path(args.emotion_model),
        emotion_map_path=_resolve_repo_path(args.emotion_map),
        config=config,
    )
    pipeline = MultimodalClassificationService(
        audio_input_service=audio_input_service,
        noise_service=noise_service,
        speech_recognition_service=speech_service,
        emotion_recognition_service=emotion_service,
    )

    if args.list_noises:
        print("\n".join(noise_service.list_available_noises()))
        return 0

    if args.list_noise_variants:
        print("\n".join(noise_service.list_available_noise_variants()))
        return 0

    if args.audio_file and args.microphone_duration is not None:
        raise SystemExit("Use either --audio-file or --microphone-duration, not both.")
    if not args.audio_file and args.microphone_duration is None:
        raise SystemExit("You must provide --audio-file or --microphone-duration.")
    if args.noise_mode == "selected" and not args.noise_type:
        raise SystemExit("When --noise-mode selected, you must provide --noise-type.")

    if args.audio_file:
        result = pipeline.process_audio_file(
            _resolve_repo_path(args.audio_file),
            noise_mode=args.noise_mode,
            noise_type=args.noise_type,
            snr_db=args.snr_db,
            speech_language=args.speech_language,
        )
    else:
        result = pipeline.process_microphone_input(
            duration=float(args.microphone_duration),
            noise_mode=args.noise_mode,
            noise_type=args.noise_type,
            snr_db=args.snr_db,
            speech_language=args.speech_language,
        )

    payload = result.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output_json:
        output_path = _resolve_repo_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
