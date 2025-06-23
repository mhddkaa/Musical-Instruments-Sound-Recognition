import librosa
import numpy as np

def extract_all_features(signal, sample_rate=44100):
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))
    rms = np.mean(librosa.feature.rms(y=signal))
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sample_rate))
    return np.hstack([mfcc_mean, zcr, rms, spec_centroid])  # Total: 16 fitur
