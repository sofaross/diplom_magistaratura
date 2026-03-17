import librosa

# ===============================
# Превращает звук в мел-спектрограмму (картинку звука)
# ===============================
def extract_mel_spectrogram(audio, sample_rate=16000):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)

    return mel_db