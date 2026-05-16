import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__" and __package__ is None:
    # Позволяет запускать файл напрямую (зелёная кнопка в PyCharm) как скрипт,
    # но при этом импортировать `src.*` как пакет.
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.evaluation.evaluation import evaluate_fusion_model
from src.inference.wav2vec2_inference import extract_embedding
from src.models.emotion_model import EmotionModel, EmotionModelImproved
from src.models.multimodal_model import FusionModel
from src.models.wav2vec2_wrapper import Wav2Vec2Wrapper
from src.utils.dataset_loader import MultimodalDataset, create_multimodal_dataloaders, prepare_splits


def _resolve_repo_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_emotion_map(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    emotion_to_id = payload.get("emotion_to_id")
    if not isinstance(emotion_to_id, dict) or not emotion_to_id:
        raise ValueError(f"Invalid emotion map JSON: {path}")
    return {str(k): int(v) for k, v in emotion_to_id.items()}


def _load_state_dict(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def _infer_emotion_set(emotion_map: dict[str, int]) -> int:
    num_classes = len(emotion_map)
    if num_classes not in {6, 8}:
        raise ValueError(
            f"Unsupported emotion map size: {num_classes}. Expected 6 or 8 classes."
        )
    return num_classes


def _load_emotion_model(checkpoint_path: Path, num_emotions: int, device: torch.device) -> torch.nn.Module:
    state = _load_state_dict(checkpoint_path)

    errors: list[str] = []
    for model_cls in (EmotionModelImproved, EmotionModel):
        try:
            model = model_cls(num_emotions=num_emotions)
            model.load_state_dict(state, strict=True)
            model.to(device)
            model.eval()
            return model
        except Exception as exc:
            errors.append(f"{model_cls.__name__}: {exc}")

    raise RuntimeError(
        "Failed to load emotion checkpoint into a supported architecture.\n"
        + "\n".join(errors)
    )


def train_fusion_model(
    fusion_model,
    speech_wrapper,
    emotion_model,
    train_loader,
    val_loader,
    epochs=5,
    lr=5e-4,
    device="cpu",
    out_dir=None,
):
    device = torch.device(device)
    fusion_model.to(device)
    emotion_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        fusion_model.train()

        running_loss = 0.0
        num_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            if len(batch) == 3:
                audios, mels, labels = batch
                lengths = None
            else:
                audios, mels, lengths, labels = batch

            mels = mels.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device) if lengths is not None else None

            with torch.inference_mode():
                speech_emb = extract_embedding(
                    speech_wrapper,
                    audios,
                    preprocess=False,
                )
                speech_emb = speech_emb.to(device)

                emotion_emb = emotion_model.extract_embedding(mels, lengths=lengths)
                emotion_emb = emotion_emb.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = fusion_model(speech_emb, emotion_emb)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            running_loss += float(loss.item()) * batch_size
            num_samples += batch_size

        train_loss = running_loss / max(1, num_samples)
        val_metrics = evaluate_fusion_model(
            fusion_model,
            speech_wrapper,
            emotion_model,
            val_loader,
            device=device,
            speech_preprocess=False,
        )
        val_acc = float(val_metrics["accuracy"])

        print(
            f"[fusion] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": fusion_model.state_dict(), "epoch": epoch, "val_metrics": val_metrics},
                out_dir / "fusion_model_last.pt",
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in fusion_model.state_dict().items()}
            if out_dir is not None:
                torch.save(
                    {"state_dict": fusion_model.state_dict(), "epoch": epoch, "val_metrics": val_metrics},
                    out_dir / "fusion_model_best.pt",
                )

    if best_state_dict is not None:
        fusion_model.load_state_dict(best_state_dict)

    return fusion_model


def main():
    parser = argparse.ArgumentParser(prog="python -m src.training.train_multimodal")
    parser.add_argument("--crema-path", default="data/raw/crema-d/AudioWAV")
    parser.add_argument("--ravdess-path", default="data/raw/ravdess")
    parser.add_argument(
        "--speech-model-name",
        default="facebook/wav2vec2-base-960h",
        help=(
            "Wav2Vec2 CTC модель из HuggingFace. "
            "Важно: для распознавания текста нужна модель, дообученная под ASR (CTC head)."
        ),
    )
    parser.add_argument("--emotion-checkpoint", default="data/processed/models/emotion/emotion_model_best.pt")
    parser.add_argument("--emotion-map-json", default="data/processed/models/emotion/emotion_map.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="data/processed/models/fusion")
    args = parser.parse_args()

    crema_path = _resolve_repo_path(args.crema_path)
    ravdess_path = _resolve_repo_path(args.ravdess_path)
    emotion_checkpoint_path = _resolve_repo_path(args.emotion_checkpoint)
    emotion_map_json_path = _resolve_repo_path(args.emotion_map_json)
    out_dir = _resolve_repo_path(args.out_dir)

    emotion_map = _load_emotion_map(emotion_map_json_path)
    emotion_set = _infer_emotion_set(emotion_map)

    try:
        train_df, val_df, test_df, _ = prepare_splits(
            crema_path,
            ravdess_path,
            emotion_map=emotion_map,
            verbose=True,
            emotion_set=emotion_set,
        )
    except ValueError as e:
        raise SystemExit(str(e))

    train_ds = MultimodalDataset(train_df)
    val_ds = MultimodalDataset(val_df)
    test_ds = MultimodalDataset(test_df)

    train_loader, val_loader, test_loader = create_multimodal_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        with_lengths=True,
    )

    device = torch.device(args.device)

    # Аудио в MultimodalDataset уже нормализовано и обрезано (trim_silence),
    # поэтому здесь можно отключить дополнительную предобработку.
    speech_wrapper = Wav2Vec2Wrapper.from_pretrained(
        model_name=args.speech_model_name,
        device=device,
    )

    emotion_model = _load_emotion_model(
        emotion_checkpoint_path,
        num_emotions=len(emotion_map),
        device=device,
    )

    fusion_model = FusionModel(
        speech_dim=speech_wrapper.hidden_size,
        emotion_dim=128,
        num_classes=len(emotion_map),
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    fusion_model = train_fusion_model(
        fusion_model,
        speech_wrapper,
        emotion_model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        out_dir=out_dir,
    )

    test_metrics = evaluate_fusion_model(
        fusion_model,
        speech_wrapper,
        emotion_model,
        test_loader,
        device=device,
        speech_preprocess=False,
    )
    print(
        f"[fusion] test_acc={test_metrics['accuracy']:.4f} "
        f"test_f1_macro={test_metrics['f1_macro']:.4f}"
    )

    torch.save(
        {"state_dict": fusion_model.state_dict(), "test_metrics": test_metrics},
        out_dir / "fusion_model_final.pt",
    )


if __name__ == "__main__":
    main()
