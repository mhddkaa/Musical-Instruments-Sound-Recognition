import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from framing_windowing import framing_windowing
from extract_all_features import extract_all_features

st.title("🎵 Pengenalan Suara Alat Musik")

scaler_path = "scaler_BestCombo.joblib"
model_path = "model_BestCombo.joblib"
scaler, model = None, None

if os.path.exists(scaler_path) and os.path.exists(model_path):
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
else:
    st.warning("❌ Model atau Scaler tidak ditemukan.")

uploaded_file = st.file_uploader("Unggah file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    signal, sr = librosa.load(uploaded_file, sr=44100)
    duration = librosa.get_duration(y=signal, sr=sr)

    if duration > 3.0:
        signal = signal[:3 * sr]
    elif duration < 3.0:
        st.warning("Audio kurang dari 3 detik, akan dipadding.")
        signal = np.pad(signal, (0, int(3 * sr - len(signal))))

    st.audio(uploaded_file, format='audio/wav')

    frames = framing_windowing(signal, sample_rate=sr)
    st.write(f"Jumlah frame: {frames.shape[0]} | Panjang tiap frame: {frames.shape[1]}")

    features = extract_all_features(signal, sample_rate=sr)
    st.write("Ekstraksi Fitur (MFCC + ZCR + RMS + Spectral Centroid):")
    st.table(features)

    st.subheader("🎶 Visualisasi MFCC")
    st.line_chart(features[:13])
    
    st.subheader("📉 Waveform")
    fig_wave, ax = plt.subplots()
    librosa.display.waveshow(signal, sr=sr, ax=ax)
    ax.set(title="Gelombang Audio (Waveform)")
    st.pyplot(fig_wave)
    
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    st.subheader("📊 Mel-Spectrogram")
    fig_spec, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Mel-Spectrogram')
    fig_spec.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig_spec)

    if scaler and model:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        st.success(f"🎼 Prediksi Alat Musik: **{prediction[0].title()}**")
    else:
        st.error("Model atau scaler belum dimuat. Tidak bisa prediksi.")