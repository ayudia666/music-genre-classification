"""
prediction.py

Modul untuk menjalankan prediksi genre per segmen, majority voting, dan
fungsi utama `predict_audio()` yang menjalankan seluruh pipeline inference
(segmentasi -> feature extraction -> feature selection -> scaling ->
prediksi -> majority voting) tanpa melakukan visualisasi apa pun.

Visualisasi (waveform, mel-spectrogram) ditangani secara terpisah oleh
modul `visualization.py`, dipanggil langsung dari `app.py`.
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.feature_extraction import extract_features, segment_audio
from utils.preprocessing import preprocess_features


def predict_segments(
    feature_matrix: np.ndarray, artifacts: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memprediksi genre untuk setiap segmen menggunakan model SVM.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix hasil `preprocess_features()`.
    artifacts : Dict[str, Any]
        Dictionary hasil `load_model()`, berisi 'model' dan 'label_encoder'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - label_numeric : hasil prediksi dalam bentuk angka
        - label_genre   : hasil prediksi dalam bentuk nama genre
    """
    model = artifacts["model"]
    label_encoder = artifacts["label_encoder"]

    label_numeric = model.predict(feature_matrix)
    label_genre = label_encoder.inverse_transform(label_numeric)

    return label_numeric, label_genre


def majority_voting(segment_predictions: List[str]) -> Dict[str, Any]:
    """
    Menghitung hasil voting mayoritas dari prediksi seluruh segmen.

    Parameters
    ----------
    segment_predictions : List[str]
        List label genre hasil prediksi untuk tiap segmen.

    Returns
    -------
    Dict[str, Any]
        Dictionary berisi:
        - 'final_genre'  : genre akhir hasil majority voting
        - 'vote_counts'  : dict jumlah vote tiap genre
        - 'vote_percent' : dict persentase vote tiap genre
    """
    vote_counts = Counter(segment_predictions)
    total_segments = len(segment_predictions)

    final_genre = vote_counts.most_common(1)[0][0]

    vote_percent = {
        genre: (count / total_segments) * 100
        for genre, count in vote_counts.items()
    }

    return {
        "final_genre": final_genre,
        "vote_counts": dict(vote_counts),
        "vote_percent": vote_percent,
    }


def predict_audio(
    y: np.ndarray, sr: int, artifacts: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Menjalankan seluruh pipeline inference pada satu sinyal audio:
    segmentasi -> feature extraction -> feature selection -> scaling ->
    prediksi per segmen -> majority voting.

    Fungsi ini TIDAK melakukan visualisasi maupun training ulang model.

    Parameters
    ----------
    y : np.ndarray
        Sinyal audio hasil loading (mono, sample rate sesuai pipeline).
    sr : int
        Sample rate sinyal audio.
    artifacts : Dict[str, Any]
        Dictionary hasil `load_model()`.

    Returns
    -------
    Dict[str, Any]
        Dictionary hasil akhir prediksi:
        - 'n_segments'          : jumlah segmen yang berhasil diproses
        - 'segment_predictions' : list genre hasil prediksi tiap segmen
        - 'vote_counts'         : jumlah vote tiap genre
        - 'vote_percent'        : persentase vote tiap genre
        - 'final_genre'         : genre akhir hasil majority voting

    Raises
    ------
    ValueError
        Jika audio terlalu pendek sehingga tidak menghasilkan satu pun
        segmen berdurasi penuh (3 detik).
    """
    # 1. Segmentasi
    segments = segment_audio(y, sr)

    if len(segments) == 0:
        raise ValueError(
            "Audio terlalu pendek untuk dibagi menjadi segmen 3 detik. "
            "Gunakan file audio dengan durasi minimal 3 detik."
        )

    # 2. Feature Extraction (per segmen)
    segment_feature_rows = [extract_features(seg, sr) for seg in segments]
    features_df = pd.DataFrame(segment_feature_rows)

    # Menjaga urutan kolom identik dengan urutan pembuatan fitur (safety check)
    features_df = features_df[list(segment_feature_rows[0].keys())]

    # 3. Feature Selection + StandardScaler
    feature_matrix = preprocess_features(features_df, artifacts)

    # 4. Prediksi Tiap Segmen
    _, label_genre = predict_segments(feature_matrix, artifacts)

    # 5. Majority Voting
    voting_result = majority_voting(list(label_genre))

    return {
        "n_segments": len(segments),
        "segment_predictions": list(label_genre),
        "vote_counts": voting_result["vote_counts"],
        "vote_percent": voting_result["vote_percent"],
        "final_genre": voting_result["final_genre"],
    }
