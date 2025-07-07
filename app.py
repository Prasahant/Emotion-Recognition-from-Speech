import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import os
import tensorflow as tf
from keras.models import load_model

# Load your trained model
model = load_model("emotion_model.h5")

# Define label mapping (adjust if different)
EMOTIONS = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

# Feature extraction function
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Web UI
st.title("🎤 Emotion Recognition from Speech")
st.markdown("Upload a `.wav` file to detect the speaker's emotion.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format='audio/wav')

    # Extract features and predict
    features = extract_features("temp.wav")
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)
    emotion = EMOTIONS.get(predicted_label, "Unknown")

    st.success(f"Predicted Emotion: **{emotion}**")
