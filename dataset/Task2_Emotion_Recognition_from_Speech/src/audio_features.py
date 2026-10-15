"""Audio loading and MFCC feature extraction utilities."""
import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MFCC = 40


def extract_features(file_path):
    """Return a fixed-length vector using the mean and std of 40 MFCCs."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {file_path}")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    means = np.mean(mfcc, axis=1)
    stds = np.std(mfcc, axis=1)
    return np.concatenate([means, stds]).astype(np.float32)
