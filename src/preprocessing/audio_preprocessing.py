import librosa
import numpy as np

# ===============================
# Загрузка аудио
# ===============================
def load_audio(path, sample_rate=16000):

    audio, sr = librosa.load(path, sr=sample_rate)

    return audio

# ===============================
# Нормализует громкость аудио
# ===============================
def normalize_audio(audio):

    audio = np.asarray(audio)

    if audio.size == 0:
        return audio

    max_abs = np.max(np.abs(audio))

    # Avoid NaNs for silent/near-silent clips.
    if not np.isfinite(max_abs) or max_abs == 0:
        return audio

    audio = audio / max_abs

    return audio

# ===============================
# Удаляет тишину в начале и в конце аудио.
# ===============================
def trim_silence(audio):

    trimmed, _ = librosa.effects.trim(audio)

    return trimmed
