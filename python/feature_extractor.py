import numpy as np
import librosa

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Spectral Centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Spectral Bandwidth
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Spectral Flatness (very important for AI detection)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=y))

    # Pitch variation
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])

    features = np.hstack([
        mfcc,
        zcr,
        centroid,
        bandwidth,
        flatness,
        rms,
        pitch
    ])

    return features
