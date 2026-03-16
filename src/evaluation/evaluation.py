import torch

from src.evaluation.metrics import compute_classification_metrics

# ===============================
# проверка модели эмоций
# ===============================
@torch.inference_mode()
def evaluate_emotion_model(model, dataloader, device="cpu"):
    model.eval()

    y_true = []
    y_pred = []

    device = torch.device(device)

    for batch in dataloader:
        # Поддерживаем два формата:
        # 1) (mels, labels) - старый
        # 2) (mels, lengths, labels) - новый (для корректного masking/padding в RNN)
        if len(batch) == 2:
            mels, labels = batch
            lengths = None
        else:
            mels, lengths, labels = batch

        mels = mels.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device) if lengths is not None else None

        logits = model(mels, lengths=lengths)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return compute_classification_metrics(y_true, y_pred)

# ===============================
# проверка мультимодальной модели
# ===============================
@torch.inference_mode()
def evaluate_fusion_model(fusion_model, speech_model, emotion_model, dataloader, device="cpu"):
    fusion_model.eval()
    emotion_model.eval()

    y_true = []
    y_pred = []

    device = torch.device(device)

    for batch in dataloader:
        # Поддерживаем (audios, mels, labels) и (audios, mels, lengths, labels)
        if len(batch) == 3:
            audios, mels, labels = batch
            lengths = None
        else:
            audios, mels, lengths, labels = batch

        mels = mels.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device) if lengths is not None else None

        speech_emb = speech_model.extract_embedding(audios)
        emotion_emb = emotion_model.extract_embedding(mels, lengths=lengths)

        logits = fusion_model(speech_emb, emotion_emb)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return compute_classification_metrics(y_true, y_pred)
