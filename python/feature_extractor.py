import numpy as np
import librosa

def extract_features(file_path):
    # ✅ Downsample + mono + float32 (huge RAM save)
    y, sr = librosa.load(file_path, sr=16000, mono=True, dtype=np.float32)

    # Trim silence (less data to process)
    y, _ = librosa.effects.trim(y)

    # ---- MFCC (light)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = np.mean(mfcc, axis=1)

    # ---- ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # ---- Spectral features (cheap)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rms = np.mean(librosa.feature.rms(y=y))

    # ❌ Removed piptrack (very heavy, not worth RAM)

    features = np.hstack([
        mfcc,
        zcr,
        centroid,
        bandwidth,
        flatness,
        rms
    ]).astype(np.float32)

    # Explicit cleanup
    del y
    del mfcc

    return features
