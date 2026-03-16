import argparse


def _smoke_check():
    import numpy as np
    import torch

    from src.preprocessing.audio_preprocessing import normalize_audio
    from src.models.emotion_model import EmotionModel
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


def main():
    parser = argparse.ArgumentParser(prog="python -m src")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small smoke-check (imports, shapes, padding).",
    )
    args = parser.parse_args()

    if args.smoke:
        _smoke_check()
        return

    parser.print_help()


if __name__ == "__main__":
    main()

