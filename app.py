"""
app.py

Aplikasi Streamlit untuk Music Genre Classification berbasis karakteristik
spektral audio (MFCC, Delta MFCC, Delta-Delta MFCC, Spectral Features,
Chroma, RMS Energy, Tempo) menggunakan model Support Vector Machine (SVM)
yang sudah dilatih pada GTZAN Music Genre Dataset.

App ini HANYA bertugas untuk:
- Menyusun layout & tampilan (UI)
- Menerima upload file audio dari pengguna
- Memanggil fungsi-fungsi pipeline machine learning yang ada di folder `utils/`
- Menampilkan hasil prediksi

Seluruh logika machine learning (feature extraction, preprocessing,
prediction) TIDAK ditulis di file ini -- lihat folder `utils/`.
"""

import base64
import os
import tempfile

import librosa
import pandas as pd
import soundfile as sf
import streamlit as st

from utils.model_loader import load_model
from utils.feature_extraction import segment_audio, extract_features, SEGMENT_DURATION, SAMPLE_RATE
from utils.preprocessing import preprocess_features
from utils.prediction import predict_segments, majority_voting
from utils.visualization import plot_waveform, plot_melspectrogram, plot_vote_bar_chart

# =========================================================
# Konfigurasi Halaman
# =========================================================
st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "models"
ASSETS_DIR = "assets"
MODEL_ACCURACY = "91.29%"
GENRE_LIST = [
    "Blues", "Classical", "Country", "Disco", "Hiphop",
    "Jazz", "Metal", "Pop", "Reggae", "Rock",
]


# =========================================================
# Utility: SVG Icon Loader
# =========================================================
def svg_img(filename: str, size: int = 20) -> str:
    path = os.path.join("assets", "icons", filename)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" style="vertical-align:middle;">'

# =========================================================
# Utility: Load Custom CSS
# =========================================================
def load_custom_css(css_path: str) -> None:
    """Menyisipkan file CSS eksternal ke dalam aplikasi Streamlit."""
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_custom_css(os.path.join(ASSETS_DIR, "style.css"))


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    logo_path = os.path.join(ASSETS_DIR, "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown(f"<div style='text-align:center;'>{svg_img('music.svg', 48)}</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-app-name' style='text-align:center;'>Music Genre Classification</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:0.85rem; color:#94A3B8;'>"
        "Identifikasi genre musik berdasarkan karakteristik spektral audio "
        "menggunakan Support Vector Machine.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-info-label'>Model</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-info-item'>Support Vector Machine (RBF)</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-info-label'>Dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-info-item'>GTZAN Music Genre Dataset</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-info-label'>Jumlah Genre</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-info-item'>10 Genre</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-info-label'>Feature Extraction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-info-item'>"
        "MFCC · Delta MFCC · Delta&sup2; MFCC<br>"
        "Spectral Features · Chroma<br>"
        "Tempo · RMS Energy"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:0.72rem; color:#64748B;'>"
        "Final Project · Audio Processing</p>",
        unsafe_allow_html=True,
    )


# =========================================================
# Hero Section
# =========================================================
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">{svg_img('music.svg', 32)} Music Genre Classification</div>
        <div class="hero-subtitle">
            Music Genre Prediction Using Spectral Audio Features and Support Vector Machine
        </div>
        <div class="hero-description">
            Unggah file audio berformat WAV atau MP3, kemudian sistem akan menganalisis
            karakteristik spektral audio untuk memprediksi genre musik menggunakan model
            Support Vector Machine (SVM) yang telah dilatih dengan GTZAN Music Genre Dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Load Model (dengan error handling)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_model_artifacts():
    """Memuat seluruh artefak model sekali saja (di-cache oleh Streamlit)."""
    return load_model(MODEL_DIR)


try:
    artifacts = get_model_artifacts()
except FileNotFoundError as e:
    st.error(
        "❌ **Gagal memuat model.**\n\n"
        f"{e}\n\n"
        "Pastikan file `svm_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, dan "
        "`selected_features.pkl` sudah ada di dalam folder `models/`."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Terjadi kesalahan tak terduga saat memuat model: {e}")
    st.stop()


# =========================================================
# Upload Audio
# =========================================================
st.markdown(f"<div class='section-title'>{svg_img('upload.svg')} Upload Audio</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop file audio di sini, atau klik untuk memilih file",
    type=["wav", "mp3"],
    help="Format yang didukung: WAV, MP3",
)

if uploaded_file is not None:
    # ----- Menyimpan file upload sementara ke disk -----
    file_suffix = os.path.splitext(uploaded_file.name)[1].lower()
    file_bytes = uploaded_file.getvalue()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(file_bytes)
            temp_path = tmp_file.name
    except Exception as e:
        st.error(f"❌ Gagal menyimpan file sementara: {e}")
        st.stop()

    # ----- Validasi & membaca metadata audio -----
    try:
        audio_info = sf.info(temp_path)
        n_channels = audio_info.channels
        native_sr = audio_info.samplerate
        duration_sec = audio_info.duration
    except Exception:
        st.error(
            "❌ **File tidak valid atau bukan file audio yang didukung.** "
            "Pastikan file berformat WAV atau MP3 dan tidak rusak."
        )
        st.stop()

    # ----- Menampilkan informasi file -----
    st.markdown(f"<div class='custom-card'><div class='section-title'>{svg_img('file.svg')} Informasi File</div>", unsafe_allow_html=True)

    info_col1, info_col2, info_col3, info_col4, info_col5 = st.columns(5)
    with info_col1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='font-size:1rem;'>{uploaded_file.name}</div>"
            f"<div class='metric-label'>Nama File</div></div>",
            unsafe_allow_html=True,
        )
    with info_col2:
        size_kb = len(file_bytes) / 1024
        size_display = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='font-size:1rem;'>{size_display}</div>"
            f"<div class='metric-label'>Ukuran File</div></div>",
            unsafe_allow_html=True,
        )
    with info_col3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='font-size:1rem;'>{duration_sec:.1f}s</div>"
            f"<div class='metric-label'>Durasi Audio</div></div>",
            unsafe_allow_html=True,
        )
    with info_col4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='font-size:1rem;'>{native_sr} Hz</div>"
            f"<div class='metric-label'>Sample Rate</div></div>",
            unsafe_allow_html=True,
        )
    with info_col5:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='font-size:1rem;'>{n_channels}</div>"
            f"<div class='metric-label'>Jumlah Channel</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ----- Audio Player -----
    st.markdown(f"<div class='custom-card'><div class='section-title'>{svg_img('audio.svg')} Audio Player</div>", unsafe_allow_html=True)
    st.audio(file_bytes)
    st.markdown("</div>", unsafe_allow_html=True)

    # ----- Warning jika durasi terlalu pendek -----
    if duration_sec < SEGMENT_DURATION:
        st.warning(
            f"⚠️ **Durasi audio terlalu pendek** ({duration_sec:.1f} detik). "
            f"Dibutuhkan minimal {int(SEGMENT_DURATION)} detik agar dapat dibagi "
            "menjadi segmen dan dianalisis. Silakan upload file audio yang lebih panjang."
        )
        st.stop()

    # ----- Loading audio untuk visualisasi & prediksi -----
    try:
        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
    except Exception as e:
        st.error(f"❌ Gagal membaca sinyal audio: {e}")
        st.stop()

    # ----- Visualisasi Audio (2 kolom) -----
    st.markdown(f"<div class='custom-card'><div class='section-title'>{svg_img('chart.svg')} Visualisasi Audio</div>", unsafe_allow_html=True)

    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        fig_wave = plot_waveform(y, sr, title="Waveform")
        st.pyplot(fig_wave, use_container_width=True)
    with viz_col2:
        fig_mel = plot_melspectrogram(y, sr, title="Mel-Spectrogram")
        st.pyplot(fig_mel, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ----- Pipeline Visualization -----
    st.markdown(
        f"""
        <div class="custom-card">
            <div class="section-title">{svg_img('pipeline.svg')} Pipeline Visualization</div>
            <div class="pipeline-container">
                <div class="pipeline-step">{svg_img('upload.svg', 16)} Upload Audio</div>
                <div class="pipeline-arrow">{svg_img('arrow.svg', 14)}</div>
                <div class="pipeline-step">{svg_img('feature.svg', 16)} Feature Extraction</div>
                <div class="pipeline-arrow">{svg_img('arrow.svg', 14)}</div>
                <div class="pipeline-step">{svg_img('selection.svg', 16)} Feature Selection</div>
                <div class="pipeline-arrow">{svg_img('arrow.svg', 14)}</div>
                <div class="pipeline-step">{svg_img('scale.svg', 16)} Scaling</div>
                <div class="pipeline-arrow">{svg_img('arrow.svg', 14)}</div>
                <div class="pipeline-step">{svg_img('svm.svg', 16)} SVM</div>
                <div class="pipeline-arrow">{svg_img('arrow.svg', 14)}</div>
                <div class="pipeline-step">{svg_img('vote.svg', 16)} Majority Voting</div>
                <div class="pipeline-arrow">{svg_img('arrow.svg', 14)}</div>
                <div class="pipeline-step">{svg_img('prediction.svg', 16)} Prediction</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Tombol Predict Genre -----
    predict_clicked = st.button("🎯 Predict Genre", use_container_width=True)

    if predict_clicked:
        try:
            # Tahap 1: Segmentasi Audio
            with st.spinner("🔄 Processing Audio..."):
                segments = segment_audio(y, sr)

            # Tahap 2: Feature Extraction + Feature Selection + Scaling
            with st.spinner("🎼 Extracting Features..."):
                segment_feature_rows = [extract_features(seg, sr) for seg in segments]
                features_df = pd.DataFrame(segment_feature_rows)
                features_df = features_df[list(segment_feature_rows[0].keys())]
                feature_matrix = preprocess_features(features_df, artifacts)

            # Tahap 3: Prediksi & Majority Voting
            with st.spinner("🎯 Predicting Genre..."):
                _, label_genre = predict_segments(feature_matrix, artifacts)
                voting_result = majority_voting(list(label_genre))

            final_genre = voting_result["final_genre"]
            vote_counts = voting_result["vote_counts"]
            vote_percent = voting_result["vote_percent"]

            # ----- Hasil Prediksi (Card Besar) -----
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">{svg_img('music.svg')} Prediction Result</div>
                    <div class="result-genre">{final_genre.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ----- Model Information (Metric Cards) -----
            st.markdown(f"<div class='custom-card'><div class='section-title'>{svg_img('info.svg')} Model Information</div>", unsafe_allow_html=True)

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{MODEL_ACCURACY}</div>"
                    f"<div class='metric-label'>Accuracy</div></div>",
                    unsafe_allow_html=True,
                )
            with mcol2:
                st.markdown(
                    "<div class='metric-card'><div class='metric-value' style='font-size:1.1rem;'>SVM</div>"
                    "<div class='metric-label'>Model</div></div>",
                    unsafe_allow_html=True,
                )
            with mcol3:
                st.markdown(
                    "<div class='metric-card'><div class='metric-value' style='font-size:1.1rem;'>GTZAN</div>"
                    "<div class='metric-label'>Dataset</div></div>",
                    unsafe_allow_html=True,
                )
            with mcol4:
                st.markdown(
                    "<div class='metric-card'><div class='metric-value'>10</div>"
                    "<div class='metric-label'>Genre Classes</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # ----- Detail Prediksi (Tabel) -----
            st.markdown(f"<div class='custom-card'><div class='section-title'>{svg_img('detail.svg')} Detail Prediksi per Segmen</div>", unsafe_allow_html=True)

            detail_df = pd.DataFrame({
                "Segment": [f"Segmen {i + 1}" for i in range(len(label_genre))],
                "Prediction": [g.capitalize() for g in label_genre],
                "Vote": ["✓" if g == final_genre else "✗" for g in label_genre],
            })
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ----- Majority Voting (Bar Chart) -----
            st.markdown(f"<div class='custom-card'><div class='section-title'>{svg_img('vote.svg')} Majority Voting</div>", unsafe_allow_html=True)
            fig_vote = plot_vote_bar_chart(vote_counts, final_genre)
            st.pyplot(fig_vote, use_container_width=True)

            vote_summary_df = pd.DataFrame({
                "Genre": list(vote_counts.keys()),
                "Jumlah Vote": list(vote_counts.values()),
                "Persentase": [f"{vote_percent[g]:.1f}%" for g in vote_counts.keys()],
            }).sort_values("Jumlah Vote", ascending=False).reset_index(drop=True)
            st.dataframe(vote_summary_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except ValueError as e:
            st.warning(f"⚠️ {e}")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses prediksi: {e}")

    # ----- Membersihkan file sementara -----
    try:
        os.remove(temp_path)
    except OSError:
        pass

else:
    st.markdown(
        f"<div class='custom-info'>{svg_img('upload.svg', 16)} Silakan upload file audio (WAV/MP3) untuk memulai prediksi genre musik.</div>",
        unsafe_allow_html=True,
    )
