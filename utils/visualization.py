import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

PRIMARY_COLOR = "#c0c1ff"
SECONDARY_COLOR = "#6bd8cb"
BG_COLOR = "#0b1326"
AXIS_COLOR = "#64748b"


def plot_waveform(y: np.ndarray, sr: int, title: str = "Waveform") -> Figure:
    matplotlib.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    librosa.display.waveshow(y, sr=sr, ax=ax, color=PRIMARY_COLOR, alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Waktu (detik)", color=AXIS_COLOR, fontsize=9)
    ax.set_ylabel("Amplitudo", color=AXIS_COLOR, fontsize=9)
    ax.tick_params(colors=AXIS_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("rgba(255,255,255,0.06)")
    fig.tight_layout()
    return fig


def plot_melspectrogram(y: np.ndarray, sr: int, title: str = "Mel-Spectrogram") -> Figure:
    matplotlib.style.use("dark_background")
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    img = librosa.display.specshow(
        mel_spec_db, sr=sr, x_axis="time", y_axis="mel",
        ax=ax, cmap="plasma",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", color="white", pad=12)
    ax.tick_params(colors=AXIS_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("rgba(255,255,255,0.06)")
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color=AXIS_COLOR, labelsize=8)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=AXIS_COLOR)
    fig.tight_layout()
    return fig


def plot_vote_bar_chart(vote_counts: dict, final_genre: str) -> Figure:
    sorted_items = sorted(vote_counts.items(), key=lambda x: -x[1])
    genre_names = [g for g, _ in sorted_items]
    genre_votes = [c for _, c in sorted_items]

    colors = [PRIMARY_COLOR if g == final_genre else "rgba(128, 131, 255, 0.2)" for g in genre_names]

    matplotlib.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    bars = ax.bar(genre_names, genre_votes, color=colors)

    for bar, count in zip(bars, genre_votes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(count),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    ax.set_ylabel("Jumlah Vote", color=AXIS_COLOR, fontsize=10)
    ax.set_title("Hasil Majority Voting per Genre", fontsize=13, fontweight="bold", color="white")
    ax.tick_params(colors=AXIS_COLOR, labelsize=9)
    plt.xticks(rotation=30, ha="right", color=AXIS_COLOR)
    for spine in ax.spines.values():
        spine.set_color("rgba(255,255,255,0.06)")
    ax.yaxis.label.set_color(AXIS_COLOR)
    fig.tight_layout()
    return fig
