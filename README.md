pip install -r requirements.txt

## Архитектура диплома

### Speech model (pretrained)

Модель: Wav2Vec2 CTC (пример: `facebook/wav2vec2-base-960h`)

Назначение:
- распознавание текста (ASR, CTC)
- извлечение **speech embeddings** из скрытых слоёв (для мультимодальности)

Код: `src/models/wav2vec2_multimodal.py` (`Wav2Vec2Multimodal`)

Пример модели для русского:
- `jonatasgrosman/wav2vec2-large-xlsr-53-russian`

### Emotion model (обучаем сами)

Архитектура:

Mel Spectrogram -> CNN -> LSTM -> Dense -> Emotion

Датасеты (лежат в `data/raw/`):

CREMA-D + RAVDESS

Код модели: `src/models/emotion_model.py` (`EmotionModel`)

Подготовка данных: `src/utils/dataset_loader.py` (mel + label)

### Multimodal fusion model

Объединяет:

speech embedding + emotion embedding -> классификатор эмоций

Код: `src/models/multimodal_model.py` (`FusionModel`)

## Запуск как пакет

Код в `src/` оформлен как Python-пакет, поэтому импорты делаются через `from src...`.

Пример сборки даталоадеров (важно: mel имеет разную длину, поэтому нужен padding-collate):

```python
from src.utils.dataset_loader import prepare_datasets, create_dataloaders

train_ds, val_ds, test_ds, emotion_map = prepare_datasets(crema_path, ravdess_path)
train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, batch_size=16)
```

Smoke-check (быстрая проверка импорта/форм):

```bash
python -m src --smoke
```

## Обучение

Emotion model:

```bash
python -m src.training.train_emotion --device cpu
```

Multimodal fusion model (после обучения emotion model):

```bash
python -m src.training.train_multimodal --device cpu
```

Извлечение speech embedding (Wav2Vec2) для одного файла:

```bash
python -m src.training.extract_speech_embedding --audio-path data/raw/test_audio/example.wav --device cpu
```

Старое имя (оставлено для совместимости):

```bash
python -m src.training.train_speech --audio-path data/raw/test_audio/example.wav --device cpu
```
