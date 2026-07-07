"""
model_loader.py

Modul untuk memuat model SVM yang sudah dilatih beserta seluruh objek
preprocessing (StandardScaler, LabelEncoder, SelectKBest) dari disk.

Modul ini TIDAK melakukan training ulang dalam bentuk apa pun -- hanya
memuat artefak (.pkl) yang sudah dihasilkan oleh notebook training.
"""

import os
from typing import Any, Dict

import joblib

# Nama file artefak model yang wajib ada di dalam MODEL_DIR
REQUIRED_MODEL_FILES = [
    "svm_model.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "selected_features.pkl",
]


def load_model(model_dir: str = "models") -> Dict[str, Any]:
    """
    Memuat seluruh artefak model yang dibutuhkan untuk inference.

    Parameters
    ----------
    model_dir : str
        Path folder yang berisi svm_model.pkl, scaler.pkl,
        label_encoder.pkl, dan selected_features.pkl.

    Returns
    -------
    Dict[str, Any]
        Dictionary berisi:
        - 'model'                  : objek SVC (model SVM terlatih)
        - 'scaler'                 : objek StandardScaler terlatih
        - 'label_encoder'          : objek LabelEncoder terlatih
        - 'selector'               : objek SelectKBest terlatih
        - 'selected_feature_names' : list nama fitur hasil seleksi

    Raises
    ------
    FileNotFoundError
        Jika salah satu file artefak model tidak ditemukan di model_dir.
    """
    missing_files = []
    for fname in REQUIRED_MODEL_FILES:
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            missing_files.append(fpath)

    if missing_files:
        raise FileNotFoundError(
            "File model berikut tidak ditemukan:\n" + "\n".join(missing_files) +
            "\n\nPastikan folder 'models/' berisi keempat file .pkl hasil training."
        )

    svm_model = joblib.load(os.path.join(model_dir, "svm_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    selected_features_obj = joblib.load(os.path.join(model_dir, "selected_features.pkl"))

    artifacts: Dict[str, Any] = {
        "model": svm_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "selector": selected_features_obj["selector"],
        "selected_feature_names": selected_features_obj["selected_feature_names"],
    }

    return artifacts
