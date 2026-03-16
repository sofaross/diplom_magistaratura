import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.evaluation.evaluation import evaluate_emotion_model
from src.models.emotion_model import EmotionModel
from src.utils.dataset_loader import create_dataloaders, prepare_datasets


def _resolve_repo_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _build_id_to_emotion(emotion_map):
    id_to_emotion = [None] * len(emotion_map)
    for emotion, idx in emotion_map.items():
        id_to_emotion[int(idx)] = emotion
    return id_to_emotion


def train_emotion_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-3,
    device="cpu",
    out_dir=None,
):
    device = torch.device(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        num_samples = 0

        for mels, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            mels = mels.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(mels)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            running_loss += float(loss.item()) * batch_size
            num_samples += batch_size

        train_loss = running_loss / max(1, num_samples)
        val_metrics = evaluate_emotion_model(model, val_loader, device=device)
        val_acc = float(val_metrics["accuracy"])

        print(
            f"[эмоция] эпоха={epoch} ошибка_обучения={train_loss:.4f} "
            f"точность_проверки={val_acc:.4f} f1_макро_проверки={val_metrics['f1_macro']:.4f}"
        )

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics},
                out_dir / "emotion_model_last.pt",
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if out_dir is not None:
                torch.save(
                    {"state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics},
                    out_dir / "emotion_model_best.pt",
                )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


def main():
    parser = argparse.ArgumentParser(prog="python -m src.training.train_emotion")
    parser.add_argument("--crema-path", default="data/raw/crema-d/AudioWAV")
    parser.add_argument("--ravdess-path", default="data/raw/ravdess")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="data/processed/models/emotion")
    args = parser.parse_args()

    crema_path = _resolve_repo_path(args.crema_path)
    ravdess_path = _resolve_repo_path(args.ravdess_path)
    out_dir = _resolve_repo_path(args.out_dir)

    try:
        train_ds, val_ds, test_ds, emotion_map = prepare_datasets(crema_path, ravdess_path)
    except ValueError as e:
        raise SystemExit(str(e))
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_emotion = _build_id_to_emotion(emotion_map)
    with open(out_dir / "emotion_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {"emotion_to_id": emotion_map, "id_to_emotion": id_to_emotion},
            f,
            ensure_ascii=False,
            indent=2,
        )

    model = EmotionModel(num_emotions=len(emotion_map))
    model = train_emotion_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        out_dir=out_dir,
    )

    test_metrics = evaluate_emotion_model(model, test_loader, device=args.device)
    print(
        f"[эмоция] точность_экзамена={test_metrics['accuracy']:.4f} "
        f"f1_макро_экзамена={test_metrics['f1_macro']:.4f}"
    )

    torch.save(
        {"state_dict": model.state_dict(), "emotion_map": emotion_map, "test_metrics": test_metrics},
        out_dir / "emotion_model_final.pt",
    )


if __name__ == "__main__":
    main()
