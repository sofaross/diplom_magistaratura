import librosa
import numpy as np


MEL_FEATURE_VERSION = "speech_mel_v2"

# Speech-oriented defaults for 16 kHz audio:
# - 25 ms window
# - 10 ms hop
# - mel range focused on the useful speech band
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 400
DEFAULT_WIN_LENGTH = 400
DEFAULT_HOP_LENGTH = 160
DEFAULT_FMIN = 20
DEFAULT_FMAX = 7600
DEFAULT_TOP_DB = 80


# ===============================
# Превращает звук в mel-спектрограмму
# ===============================
def extract_mel_spectrogram(
    audio,
    sample_rate=16000,
    *,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    win_length: int = DEFAULT_WIN_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: int = DEFAULT_FMIN,
    fmax: int = DEFAULT_FMAX,
    top_db: int = DEFAULT_TOP_DB,
):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=int(sample_rate),
        n_fft=int(n_fft),
        win_length=int(win_length),
        hop_length=int(hop_length),
        n_mels=int(n_mels),
        fmin=float(fmin),
        fmax=float(fmax),
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=float(top_db))
    return mel_db
