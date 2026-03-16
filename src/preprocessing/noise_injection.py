import numpy as np


def add_white_noise(audio, noise_factor=0.005):

    noise = np.random.randn(len(audio))

    augmented = audio + noise_factor * noise

    return augmented