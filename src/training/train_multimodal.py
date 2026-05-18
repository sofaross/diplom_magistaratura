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

from src.evaluation.evaluation import evaluate_fusion_model
from src.inference.wav2vec2_inference import extract_embedding
from src.models.emotion_model import EmotionModel, EmotionModelImproved
from src.models.multimodal_model import FusionModel
from src.models.wav2vec2_wrapper import Wav2Vec2Wrapper
from src.utils.dataset_loader import (
    MelAugmentConfig,
    WaveformNoiseAugmentConfig,
    create_multimodal_dataloaders,
    prepare_multimodal_datasets,
)


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
        raise ValueError(f"Unsupported emotion map size: {num_classes}. Expected 6 or 8 classes.")
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
    epochs=20,
    lr=5e-4,
    weight_decay=5e-4,
    max_grad_norm=1.0,
    *,
    scheduler_name: str = "onecycle",
    onecycle_pct_start: float = 0.1,
    early_stopping_patience=6,
    early_stopping_min_delta=1e-4,
    speech_layer: int = -1,
    speech_pool: str = "mean",
    device="cpu",
    out_dir=None,
):
    device = torch.device(device)
    fusion_model.to(device)
    emotion_model.to(device)
    emotion_model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(fusion_model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    scheduler = None
    scheduler_name = str(scheduler_name).lower().strip()
    if scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            threshold=1e-4,
        )
    elif scheduler_name == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(lr),
            epochs=int(epochs),
            steps_per_epoch=len(train_loader),
            pct_start=float(onecycle_pct_start),
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
    elif scheduler_name != "none":
        raise ValueError(f"Unknown scheduler_name={scheduler_name!r}. Use 'none', 'plateau' or 'onecycle'.")

    best_val_acc = -1.0
    best_val_f1 = -1.0
    best_epoch = 0
    best_state_dict = None
    bad_epochs = 0

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
                    layer=int(speech_layer),
                    pool=str(speech_pool),
                    preprocess=False,
                ).to(device)
                emotion_emb = emotion_model.extract_embedding(mels, lengths=lengths).to(device)

            speech_emb = speech_emb.detach().clone()
            emotion_emb = emotion_emb.detach().clone()

            optimizer.zero_grad(set_to_none=True)

            logits = fusion_model(speech_emb, emotion_emb)
            loss = criterion(logits, labels)

            loss.backward()
            if max_grad_norm is not None and float(max_grad_norm) > 0:
                nn.utils.clip_grad_norm_(fusion_model.parameters(), float(max_grad_norm))
            optimizer.step()
            if scheduler_name == "onecycle":
                scheduler.step()

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
            speech_layer=int(speech_layer),
            speech_pool=str(speech_pool),
        )
        val_acc = float(val_metrics["accuracy"])
        val_f1 = float(val_metrics["f1_macro"])

        if scheduler_name == "plateau":
            scheduler.step(val_f1)

        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[fusion] epoch={epoch} lr={current_lr:.2e} train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} val_f1_macro={val_f1:.4f}"
        )

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": fusion_model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "speech_layer": int(speech_layer),
                    "speech_pool": str(speech_pool),
                },
                out_dir / "fusion_model_last.pt",
            )

        if val_f1 > best_val_f1 + float(early_stopping_min_delta):
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = int(epoch)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in fusion_model.state_dict().items()}
            bad_epochs = 0
            if out_dir is not None:
                torch.save(
                    {
                        "state_dict": fusion_model.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "speech_layer": int(speech_layer),
                        "speech_pool": str(speech_pool),
                    },
                    out_dir / "fusion_model_best.pt",
                )
        else:
            bad_epochs += 1
            if bad_epochs >= int(early_stopping_patience):
                print(
                    f"[fusion] early stopping: нет улучшения val_f1_macro "
                    f"{bad_epochs} эпох подряд (patience={early_stopping_patience})."
                )
                break

    if best_state_dict is not None:
        fusion_model.load_state_dict(best_state_dict)

    training_summary = {
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_acc),
        "best_val_f1_macro": float(best_val_f1),
        "selection_metric": "val_f1_macro",
        "stopped_epoch": int(epoch),
        "speech_layer": int(speech_layer),
        "speech_pool": str(speech_pool),
    }
    return fusion_model, training_summary


def main():
    parser = argparse.ArgumentParser(prog="python -m src.training.train_multimodal")
    parser.add_argument("--crema-path", default="data/raw/crema-d/AudioWAV")
    parser.add_argument("--ravdess-path", default="data/raw/ravdess")
    parser.add_argument("--emotion-set", type=int, choices=[6, 8], default=6)
    parser.add_argument(
        "--speech-model-name",
        default="facebook/wav2vec2-base-960h",
        help=(
            "Wav2Vec2 CTC модель из HuggingFace. "
            "Для fusion используется её acoustic embedding, а не распознанный текст."
        ),
    )
    parser.add_argument("--speech-layer", type=int, default=-1, help="Слой speech-модели для embedding (-1 = последний).")
    parser.add_argument("--speech-pool", choices=["mean", "max"], default="mean", help="Пулинг по времени для speech embedding.")
    parser.add_argument("--emotion-checkpoint", default="data/processed/models/emotion/emotion_model_final.pt")
    parser.add_argument("--emotion-map-json", default="data/processed/models/emotion/emotion_map.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["none", "plateau", "onecycle"], default="onecycle")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="data/processed/models/fusion")
    parser.add_argument("--cache-dir", default="data/processed/features/mel_cache")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Если >0, ограничивает длину mel по времени для emotion-ветки fusion-модели.",
    )
    parser.add_argument("--noise-std-max", type=float, default=0.25)
    parser.add_argument("--time-stretch-min", type=float, default=0.85)
    parser.add_argument("--time-stretch-max", type=float, default=1.20)
    parser.add_argument("--pitch-shift-bins", type=int, default=6)
    parser.add_argument("--waveform-noise-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-prob", type=float, default=0.5)
    parser.add_argument("--noise-dir", default="data/noise")
    parser.add_argument("--noise-types", nargs="+", default=["white", "pink", "brown", "real"])
    parser.add_argument("--snr-min", type=float, default=5.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    args = parser.parse_args()

    crema_path = _resolve_repo_path(args.crema_path)
    ravdess_path = _resolve_repo_path(args.ravdess_path)
    emotion_checkpoint_path = _resolve_repo_path(args.emotion_checkpoint)
    emotion_map_json_path = _resolve_repo_path(args.emotion_map_json)
    out_dir = _resolve_repo_path(args.out_dir)
    cache_dir = _resolve_repo_path(args.cache_dir)
    noise_dir = _resolve_repo_path(args.noise_dir)

    emotion_map = _load_emotion_map(emotion_map_json_path)
    emotion_set = int(args.emotion_set)
    checkpoint_emotion_set = _infer_emotion_set(emotion_map)
    if checkpoint_emotion_set != emotion_set:
        raise SystemExit(
            f"emotion-set={emotion_set} не совпадает с emotion_map из чекпоинта "
            f"({checkpoint_emotion_set} классов)."
        )

    try:
        aug_cfg = MelAugmentConfig(
            noise_std_max=float(args.noise_std_max),
            time_stretch_min=float(args.time_stretch_min),
            time_stretch_max=float(args.time_stretch_max),
            pitch_shift_bins=int(args.pitch_shift_bins),
        )
        waveform_noise_cfg = None
        if bool(args.waveform_noise_augment):
            waveform_noise_cfg = WaveformNoiseAugmentConfig(
                noise_prob=float(args.noise_prob),
                noise_dir=noise_dir,
                noise_types=tuple(str(name) for name in args.noise_types),
                snr_min=float(args.snr_min),
                snr_max=float(args.snr_max),
            )
            print("[fusion] Включена waveform-level noise augmentation.")
            print(
                f"[fusion] noise_prob={float(args.noise_prob):.2f} "
                f"noise_types={', '.join(str(name) for name in args.noise_types)} "
                f"snr=[{float(args.snr_min):g}, {float(args.snr_max):g}] dB "
                f"noise_dir={noise_dir}"
            )
            print("[fusion] Валидация и тест останутся без шумовой аугментации.")
        else:
            print("[fusion] Waveform-level noise augmentation отключена.")

        train_ds, val_ds, test_ds, _ = prepare_multimodal_datasets(
            crema_path,
            ravdess_path,
            emotion_map=emotion_map,
            emotion_set=emotion_set,
            augment=bool(args.augment),
            augment_config=aug_cfg,
            waveform_noise_augment=bool(args.waveform_noise_augment),
            waveform_noise_config=waveform_noise_cfg,
            max_frames=(int(args.max_frames) if int(args.max_frames) > 0 else None),
            cache_dir=cache_dir,
        )
        if bool(args.waveform_noise_augment):
            noise_manager = getattr(train_ds, "noise_manager", None)
            load_errors = dict(getattr(noise_manager, "load_errors", {})) if noise_manager is not None else {}
            if load_errors:
                print(f"[fusion] Отброшено шумовых файлов: {len(load_errors)}")
                print("[fusion] Отброшенные шумовые файлы:")
                for bad_path, reason in sorted(load_errors.items()):
                    print(f"  - {Path(bad_path).name}: {reason}")
            else:
                print("[fusion] Отброшенных шумовых файлов: 0")
    except ValueError as e:
        raise SystemExit(str(e))

    train_loader, val_loader, test_loader = create_multimodal_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_sampling=bool(args.balanced_sampling),
        with_lengths=True,
    )

    device = torch.device(args.device)

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

    fusion_model, training_summary = train_fusion_model(
        fusion_model,
        speech_wrapper,
        emotion_model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        scheduler_name=args.scheduler,
        onecycle_pct_start=args.onecycle_pct_start,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        speech_layer=args.speech_layer,
        speech_pool=args.speech_pool,
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
        speech_layer=args.speech_layer,
        speech_pool=args.speech_pool,
    )
    print(
        f"[fusion] test_acc={test_metrics['accuracy']:.4f} "
        f"test_f1_macro={test_metrics['f1_macro']:.4f}"
    )
    print(
        f"[fusion] Лучшая эпоха: {training_summary['best_epoch']} "
        f"(val_acc={training_summary['best_val_accuracy']:.4f}, "
        f"val_f1_macro={training_summary['best_val_f1_macro']:.4f})"
    )
    print("[fusion] Финальная модель содержит лучшие веса по val_f1_macro.")

    torch.save(
        {
            "state_dict": fusion_model.state_dict(),
            "test_metrics": test_metrics,
            "training_summary": training_summary,
            "speech_layer": int(args.speech_layer),
            "speech_pool": str(args.speech_pool),
        },
        out_dir / "fusion_model_final.pt",
    )


if __name__ == "__main__":
    main()
