import torch

from src.evaluation.metrics import compute_classification_metrics


@torch.inference_mode()
def evaluate_emotion_model(model, dataloader, device="cpu"):
    model.eval()

    y_true = []
    y_pred = []

    device = torch.device(device)

    for mels, labels in dataloader:
        mels = mels.to(device)
        labels = labels.to(device)

        logits = model(mels)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return compute_classification_metrics(y_true, y_pred)


@torch.inference_mode()
def evaluate_fusion_model(fusion_model, speech_model, emotion_model, dataloader, device="cpu"):
    fusion_model.eval()
    emotion_model.eval()

    y_true = []
    y_pred = []

    device = torch.device(device)

    for audios, mels, labels in dataloader:
        mels = mels.to(device)
        labels = labels.to(device)

        speech_emb = speech_model.extract_embedding(audios)
        emotion_emb = emotion_model.extract_embedding(mels)

        logits = fusion_model(speech_emb, emotion_emb)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return compute_classification_metrics(y_true, y_pred)
