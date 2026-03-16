import torch

from src.features.feature_extraction import extract_mel_spectrogram
from src.preprocessing.audio_preprocessing import load_audio, normalize_audio, trim_silence


def load_audio_for_models(path, sample_rate=16000):
    audio = load_audio(path, sample_rate=sample_rate)
    audio = normalize_audio(audio)
    audio = trim_silence(audio)
    return audio


def audio_to_mel_tensor(audio):
    mel = extract_mel_spectrogram(audio)
    mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()
    return mel


def load_audio_and_mel(path, sample_rate=16000):
    audio = load_audio_for_models(path, sample_rate=sample_rate)
    mel = audio_to_mel_tensor(audio)
    return audio, mel
