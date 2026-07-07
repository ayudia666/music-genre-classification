"""
preprocessing.py

Modul untuk menerapkan feature selection dan standardisasi fitur
menggunakan objek yang SUDAH DILATIH sebelumnya (selector dan scaler).

PENTING: Modul ini hanya memanggil `transform()`, TIDAK PERNAH `fit()`.
Objek `selector` (SelectKBest) dan `scaler` (StandardScaler) sudah
"belajar" dari data training; memanggil fit() di sini akan merusak
konsistensi pipeline dan membuat hasil prediksi tidak valid.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def preprocess_features(features_df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
    """
    Menerapkan feature selection dan standardisasi pada fitur hasil ekstraksi.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame fitur mentah (kolom harus sama persis dengan saat training).
    artifacts : Dict[str, Any]
        Dictionary hasil `load_model()`, harus berisi key 'selector' dan 'scaler'.

    Returns
    -------
    np.ndarray
        Feature matrix yang siap digunakan untuk prediksi (sudah diseleksi
        dan distandardisasi).
    """
    selector = artifacts["selector"]
    scaler = artifacts["scaler"]

    # Feature Selection (transform saja, TIDAK fit)
    features_selected = selector.transform(features_df)

    # Standardisasi (transform saja, TIDAK fit)
    feature_matrix = scaler.transform(features_selected)

    return feature_matrix
