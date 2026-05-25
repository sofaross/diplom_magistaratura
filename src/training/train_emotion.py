import sys
from pathlib import Path

import argparse
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluation import evaluate_emotion_model
from src.models.emotion_model import EmotionModel, EmotionModelImproved
from src.training.losses import FocalCrossEntropyLoss, build_inverse_frequency_class_weights
from src.utils.dataset_loader import (
    MelAugmentConfig,
    WaveformNoiseAugmentConfig,
    create_dataloaders,
    prepare_datasets,
)


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
    weight_decay=1e-2,
    label_smoothing=0.05,
    max_grad_norm=1.0,
    *,
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    scheduler_name: str = "plateau",
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    onecycle_pct_start: float = 0.1,
    early_stopping_patience=8,
    early_stopping_min_delta=1e-4,
    use_class_weights: bool = True,
    device="cpu",
    out_dir=None,
):
    device = torch.device(device)
    model.to(device)

    class_weights = None
    if bool(use_class_weights):
        try:
            labels = train_loader.dataset.df["label"].tolist()
            class_weights = build_inverse_frequency_class_weights(labels)
        except Exception:
            class_weights = None

    loss_name = str(loss_name).lower().strip()
    loss_weights = class_weights.to(device) if class_weights is not None else None
    if loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(
            weight=loss_weights,
            label_smoothing=float(label_smoothing),
        )
    elif loss_name == "focal":
        criterion = FocalCrossEntropyLoss(
            gamma=float(focal_gamma),
            weight=loss_weights,
            label_smoothing=float(label_smoothing),
            reduction="mean",
        )
    else:
        raise ValueError(f"Unknown loss_name={loss_name!r}. Use 'cross_entropy' or 'focal'.")

    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
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
    else:
        raise ValueError(f"Unknown scheduler_name={scheduler_name!r}. Use 'plateau' or 'onecycle'.")

    best_val_acc = -1.0
    best_val_f1 = -1.0
    best_epoch = 0
    best_state_dict = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            if len(batch) == 2:
                mels, labels = batch
                lengths = None
            else:
                mels, lengths, labels = batch

            mels = mels.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device) if lengths is not None else None

            optimizer.zero_grad(set_to_none=True)

            do_mixup = (
                float(mixup_alpha) > 0
                and float(mixup_prob) > 0
                and random.random() < float(mixup_prob)
            )
            if do_mixup and int(labels.shape[0]) > 1:
                lam = float(
                    torch.distributions.Beta(float(mixup_alpha), float(mixup_alpha)).sample().item()
                )
                index = torch.randperm(int(labels.shape[0]), device=device)
                mixed_mels = lam * mels + (1.0 - lam) * mels[index]
                y_a = labels
                y_b = labels[index]

                mixed_lengths = None
                if lengths is not None:
                    mixed_lengths = torch.maximum(lengths, lengths[index])

                logits = model(mixed_mels, lengths=mixed_lengths)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                logits = model(mels, lengths=lengths)
                loss = criterion(logits, labels)

            loss.backward()
            if max_grad_norm is not None and float(max_grad_norm) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()
            if scheduler_name == "onecycle":
                scheduler.step()

            batch_size = int(labels.shape[0])
            running_loss += float(loss.item()) * batch_size
            num_samples += batch_size

        train_loss = running_loss / max(1, num_samples)
        val_metrics = evaluate_emotion_model(model, val_loader, device=device)
        val_acc = float(val_metrics["accuracy"])
        val_f1 = float(val_metrics["f1_macro"])

        if scheduler_name == "plateau":
            scheduler.step(val_f1)

        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[эмоция] эпоха={epoch} lr={current_lr:.2e} ошибка_обучения={train_loss:.4f} "
            f"точность_проверки={val_acc:.4f} f1_макро_проверки={val_f1:.4f}"
        )

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "loss_name": loss_name,
                    "focal_gamma": float(focal_gamma),
                },
                out_dir / "emotion_model_last.pt",
            )

        if val_f1 > best_val_f1 + float(early_stopping_min_delta):
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = int(epoch)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(early_stopping_patience):
                print(
                    f"[эмоция] early stopping: нет улучшения val_f1_macro "
                    f"{bad_epochs} эпох подряд (patience={early_stopping_patience})."
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    training_summary = {
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_acc),
        "best_val_f1_macro": float(best_val_f1),
        "selection_metric": "val_f1_macro",
        "loss_name": loss_name,
        "focal_gamma": float(focal_gamma),
        "use_class_weights": bool(use_class_weights),
        "stopped_epoch": int(epoch),
    }
    return model, training_summary


def main():
    parser = argparse.ArgumentParser(prog="python -m src.training.train_emotion")
    parser.add_argument("--crema-path", default="data/raw/crema-d/AudioWAV")
    parser.add_argument("--ravdess-path", default="data/raw/ravdess")
    parser.add_argument("--model", choices=["baseline", "improved"], default="improved")
    parser.add_argument("--emotion-set", type=int, choices=[6, 8], default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--scheduler", choices=["plateau", "onecycle"], default="onecycle")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="data/processed/models/emotion")
    parser.add_argument("--cache-dir", default="data/processed/features/mel_cache")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames",type=int,default=0,help="Если >0, ограничивает длину mel по времени (train=random crop, val/test=center crop).",)
    parser.add_argument("--noise-std-max", type=float, default=0.25)
    parser.add_argument("--time-stretch-min", type=float, default=0.85)
    parser.add_argument("--time-stretch-max", type=float, default=1.20)
    parser.add_argument("--pitch-shift-bins", type=int, default=6)
    parser.add_argument("--waveform-noise-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-prob", type=float, default=0.4)
    parser.add_argument("--noise-dir", default="data/noise")
    parser.add_argument("--noise-types", nargs="+", default=["white", "pink", "brown", "real"])
    parser.add_argument("--snr-min", type=float, default=8.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--use-resd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resd-mode", choices=["train_only", "full_mix"], default="full_mix")
    parser.add_argument("--resd-dataset-name", default="Aniemore/resd")
    parser.add_argument("--resd-splits", nargs="+", default=["train"])
    parser.add_argument("--quality-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-filter-iqr-multiplier", type=float, default=2.0)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    crema_path = _resolve_repo_path(args.crema_path)
    ravdess_path = _resolve_repo_path(args.ravdess_path)
    out_dir = _resolve_repo_path(args.out_dir)
    cache_dir = _resolve_repo_path(args.cache_dir)
    noise_dir = _resolve_repo_path(args.noise_dir)

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
            print("[эмоция] Включена waveform-level noise augmentation.")
            print(
                f"[эмоция] noise_prob={float(args.noise_prob):.2f} "
                f"noise_types={', '.join(str(name) for name in args.noise_types)} "
                f"snr=[{float(args.snr_min):g}, {float(args.snr_max):g}] dB "
                f"noise_dir={noise_dir}"
            )
            print("[эмоция] Валидация и тест останутся без шумовой аугментации.")
        else:
            print("[эмоция] Waveform-level noise augmentation отключена.")

        if bool(args.use_resd):
            print(
                f"[эмоция] Подключён дополнительный датасет {args.resd_dataset_name} "
                f"со split: {', '.join(str(name) for name in args.resd_splits)}."
            )
            if str(args.resd_mode) == "train_only":
                print("[эмоция] RESD будет добавлен только в train после базового split.")
            else:
                print("[эмоция] RESD будет полностью смешан с CREMA-D и RAVDESS до разбиения на train/val/test.")
            print("[эмоция] Emotion 'enthusiasm' будет исключена, остальные 6 эмоций будут приведены к схеме проекта.")
        if bool(args.quality_filter):
            print(
                f"[эмоция] Включена фильтрация аномальных записей по качеству "
                f"(boxplot/IQR, множитель={float(args.quality_filter_iqr_multiplier):g})."
            )
        else:
            print("[эмоция] Фильтрация аномальных записей по качеству отключена.")

        train_ds, val_ds, test_ds, emotion_map = prepare_datasets(
            crema_path,
            ravdess_path,
            emotion_set=int(args.emotion_set),
            augment=bool(args.augment),
            augment_config=aug_cfg,
            waveform_noise_augment=bool(args.waveform_noise_augment),
            waveform_noise_config=waveform_noise_cfg,
            max_frames=(int(args.max_frames) if int(args.max_frames) > 0 else None),
            cache_dir=cache_dir,
            use_resd=bool(args.use_resd),
            resd_mode=str(args.resd_mode),
            resd_dataset_name=str(args.resd_dataset_name),
            resd_splits=tuple(str(name) for name in args.resd_splits),
            quality_filter=bool(args.quality_filter),
            quality_filter_iqr_multiplier=float(args.quality_filter_iqr_multiplier),
        )

        if bool(args.waveform_noise_augment):
            noise_manager = getattr(train_ds, "noise_manager", None)
            load_errors = dict(getattr(noise_manager, "load_errors", {})) if noise_manager is not None else {}
            if load_errors:
                print(f"[эмоция] Отброшено шумовых файлов: {len(load_errors)}")
                print("[эмоция] Отброшенные шумовые файлы:")
                for bad_path, reason in sorted(load_errors.items()):
                    print(f"  - {Path(bad_path).name}: {reason}")
            else:
                print("[эмоция] Отброшенных шумовых файлов: 0")
    except ValueError as exc:
        raise SystemExit(str(exc))

    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_sampling=bool(args.balanced_sampling),
        with_lengths=True,
        pad_value=0.0,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_emotion = _build_id_to_emotion(emotion_map)
    with open(out_dir / "emotion_map.json", "w", encoding="utf-8") as file:
        json.dump(
            {"emotion_to_id": emotion_map, "id_to_emotion": id_to_emotion},
            file,
            ensure_ascii=False,
            indent=2,
        )

    if args.model == "baseline":
        model = EmotionModel(num_emotions=len(emotion_map))
    else:
        model = EmotionModelImproved(num_emotions=len(emotion_map))

    print(f"[эмоция] Архитектура модели: {model.__class__.__name__}")
    if str(args.loss) == "focal":
        print(f"[эмоция] Используется focal loss (gamma={float(args.focal_gamma):.2f}).")
    else:
        print("[эмоция] Используется cross-entropy loss.")

    print(f"[эмоция] Class weights: {'enabled' if bool(args.class_weights) else 'disabled'}.")

    model, training_summary = train_emotion_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        scheduler_name=args.scheduler,
        onecycle_pct_start=args.onecycle_pct_start,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        use_class_weights=bool(args.class_weights),
        device=args.device,
        out_dir=out_dir,
    )

    test_metrics = evaluate_emotion_model(model, test_loader, device=args.device)
    print(
        f"[эмоция] точность_экзамена={test_metrics['accuracy']:.4f} "
        f"f1_макро_экзамена={test_metrics['f1_macro']:.4f}"
    )

    final_checkpoint_path = out_dir / "emotion_model_final.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "emotion_map": emotion_map,
            "test_metrics": test_metrics,
            "training_summary": training_summary,
        },
        final_checkpoint_path,
    )
    print(
        f"[эмоция] Лучшая эпоха: {training_summary['best_epoch']} "
        f"(val_acc={training_summary['best_val_accuracy']:.4f}, "
        f"val_f1_macro={training_summary['best_val_f1_macro']:.4f})"
    )
    print("[эмоция] Финальная модель содержит лучшие веса по val_f1_macro.")
    print(f"[эмоция] Финальная архитектура модели: {model.__class__.__name__}")
    print(f"[эмоция] Финальный чекпоинт сохранён: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
