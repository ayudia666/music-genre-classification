"""
feature_extraction.py

Modul untuk segmentasi audio dan ekstraksi fitur spektral.

PENTING: Fitur, urutan kolom, dan parameter pada modul ini HARUS identik
dengan notebook training maupun notebook Model Inference. Jangan mengubah
urutan atau nama fitur, karena objek `selector` (SelectKBest) dan `scaler`
(StandardScaler) yang sudah dilatih mengharapkan urutan kolom yang sama
persis seperti saat training.
"""

from typing import Dict, List

import librosa
import numpy as np

# Konstanta pipeline -- HARUS sama persis dengan notebook training/inference
SEGMENT_DURATION: float = 3.0   # detik
SAMPLE_RATE: int = 22050
N_MFCC: int = 20


def segment_audio(
    y: np.ndarray,
    sr: int,
    segment_duration: float = SEGMENT_DURATION,
) -> List[np.ndarray]:
    """
    Memecah sinyal audio menjadi beberapa segmen dengan durasi tetap.

    Sisa sinyal yang lebih pendek dari `segment_duration` di ujung file
    diabaikan, agar seluruh segmen memiliki panjang yang konsisten.

    Parameters
    ----------
    y : np.ndarray
        Sinyal audio hasil `librosa.load()`.
    sr : int
        Sample rate sinyal audio.
    segment_duration : float
        Durasi tiap segmen dalam detik (default: 3 detik).

    Returns
    -------
    List[np.ndarray]
        List berisi segmen-segmen audio dengan panjang yang sama.
    """
    segment_length = int(segment_duration * sr)
    n_segments = len(y) // segment_length

    segments = [
        y[i * segment_length: (i + 1) * segment_length]
        for i in range(n_segments)
    ]
    return segments


def extract_features(y_segment: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> Dict[str, float]:
    """
    Mengekstrak fitur spektral dari satu segmen audio.

    Fitur yang diekstrak (masing-masing diringkas menjadi mean & std):
    MFCC (20) -> Delta MFCC -> Delta-Delta MFCC -> Spectral Centroid ->
    Spectral Bandwidth -> Spectral Roll-off -> Spectral Contrast ->
    Zero Crossing Rate -> Chroma STFT -> RMS Energy -> Tempo.

    Parameters
    ----------
    y_segment : np.ndarray
        Sinyal audio untuk satu segmen (durasi tetap).
    sr : int
        Sample rate sinyal.
    n_mfcc : int
        Jumlah koefisien MFCC (default: 20, harus sama dengan training).

    Returns
    -------
    Dict[str, float]
        Dictionary berisi nama fitur (mean/std) dan nilainya.
    """
    features: Dict[str, float] = {}

    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    # 2. Delta MFCC
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    for i in range(n_mfcc):
        features[f"delta_mfcc_{i + 1}_mean"] = float(np.mean(delta_mfcc[i]))
        features[f"delta_mfcc_{i + 1}_std"] = float(np.std(delta_mfcc[i]))

    # 3. Delta-Delta MFCC
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for i in range(n_mfcc):
        features[f"delta2_mfcc_{i + 1}_mean"] = float(np.mean(delta2_mfcc[i]))
        features[f"delta2_mfcc_{i + 1}_std"] = float(np.std(delta2_mfcc[i]))

    # 4. Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(spec_centroid))
    features["spectral_centroid_std"] = float(np.std(spec_centroid))

    # 5. Spectral Bandwidth
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)
    features["spectral_bandwidth_mean"] = float(np.mean(spec_bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(spec_bandwidth))

    # 6. Spectral Roll-off
    spec_rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(spec_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spec_rolloff))

    # 7. Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y_segment, sr=sr)
    for i in range(spec_contrast.shape[0]):
        features[f"spectral_contrast_{i + 1}_mean"] = float(np.mean(spec_contrast[i]))
        features[f"spectral_contrast_{i + 1}_std"] = float(np.std(spec_contrast[i]))

    # 8. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y_segment)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # 9. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr)
    for i in range(chroma.shape[0]):
        features[f"chroma_{i + 1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{i + 1}_std"] = float(np.std(chroma[i]))

    # 10. RMS Energy
    rms = librosa.feature.rms(y=y_segment)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # 11. Tempo
    try:
        tempo_frames = librosa.feature.tempo(y=y_segment, sr=sr, aggregate=None)
        features["tempo_mean"] = float(np.mean(tempo_frames))
        features["tempo_std"] = float(np.std(tempo_frames))
    except Exception:
        tempo_value = librosa.feature.tempo(y=y_segment, sr=sr, aggregate=np.mean)
        features["tempo_mean"] = float(np.atleast_1d(tempo_value)[0])
        features["tempo_std"] = 0.0

    return features
