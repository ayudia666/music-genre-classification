"""
visualization.py

Modul untuk membuat visualisasi audio (waveform dan mel-spectrogram)
menggunakan matplotlib. Fungsi-fungsi di sini mengembalikan objek
`matplotlib.figure.Figure`, sehingga dapat ditampilkan langsung oleh
`app.py` menggunakan `st.pyplot(fig)`.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Palet warna aplikasi, agar visualisasi konsisten dengan tema UI
PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#06B6D4"


def plot_waveform(y: np.ndarray, sr: int, title: str = "Waveform") -> Figure:
    """
    Membuat plot waveform (amplitudo terhadap waktu) dari sinyal audio.

    Parameters
    ----------
    y : np.ndarray
        Sinyal audio.
    sr : int
        Sample rate sinyal audio.
    title : str
        Judul plot.

    Returns
    -------
    matplotlib.figure.Figure
        Objek figure yang siap ditampilkan dengan st.pyplot().
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color=PRIMARY_COLOR)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Amplitudo")
    fig.tight_layout()
    return fig


def plot_melspectrogram(y: np.ndarray, sr: int, title: str = "Mel-Spectrogram") -> Figure:
    """
    Membuat plot mel-spectrogram dari sinyal audio.

    Parameters
    ----------
    y : np.ndarray
        Sinyal audio.
    sr : int
        Sample rate sinyal audio.
    title : str
        Judul plot.

    Returns
    -------
    matplotlib.figure.Figure
        Objek figure yang siap ditampilkan dengan st.pyplot().
    """
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="viridis")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_vote_bar_chart(vote_counts: dict, final_genre: str) -> Figure:
    """
    Membuat bar chart hasil majority voting antar genre.

    Genre dengan hasil akhir (final_genre) ditonjolkan dengan warna
    berbeda dibandingkan genre lainnya.

    Parameters
    ----------
    vote_counts : dict
        Dictionary {nama_genre: jumlah_vote}.
    final_genre : str
        Genre hasil akhir majority voting (akan ditonjolkan warnanya).

    Returns
    -------
    matplotlib.figure.Figure
        Objek figure yang siap ditampilkan dengan st.pyplot().
    """
    sorted_items = sorted(vote_counts.items(), key=lambda x: -x[1])
    genre_names = [g for g, _ in sorted_items]
    genre_votes = [c for _, c in sorted_items]

    colors = ["#22C55E" if g == final_genre else "#CBD5E1" for g in genre_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(genre_names, genre_votes, color=colors)

    for bar, count in zip(bars, genre_votes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(count),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Jumlah Vote")
    ax.set_title("Hasil Majority Voting per Genre", fontsize=12, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig
