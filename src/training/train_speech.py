import argparse
from pathlib import Path

import torch

from src.models.speech_model import SpeechEmbeddingModel
from src.utils.audio_utils import load_audio_for_models


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src.training.train_speech",
        description="Speech model is pretrained; this script extracts speech embeddings.",
    )
    parser.add_argument("--audio-path", required=True, help="Path to a .wav file.")
    parser.add_argument("--model-name", default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="Optional path to save embedding as .pt")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    audio = load_audio_for_models(audio_path, sample_rate=16000)

    model = SpeechEmbeddingModel(model_name=args.model_name, device=args.device)
    embedding = model.extract_embedding(audio)

    print(f"speech embedding shape: {tuple(embedding.shape)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding.detach().cpu(), out_path)


if __name__ == "__main__":
    main()
