import streamlit as st
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMClassifier

# -----------------------------
# Load model and scaler
# -----------------------------
with open("Spotify_LGBM_Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Spotify_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="🎵 Moodify: Spotify Song Emotion Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Spotify theme
# -----------------------------
st.markdown("""
<style>
    .main { background-color: #191414; color: #ffffff; }
    .stButton>button { background-color: #1DB954; color: white; font-weight: bold; border-radius: 10px; }
    .sidebar .sidebar-content { background-color: #121212; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# App title
# -----------------------------
st.title("🎵 Moodify: Spotify Song Emotion Predictor")
st.markdown("Predict the emotion of a Spotify track based on its audio features.")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Input Audio Features")

def user_input_features():
    duration = st.sidebar.number_input("Duration (ms)", 1000, 4000000, 200000)
    danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.sidebar.slider("Loudness (dB)", -60.0, 5.0, -10.0)
    speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.05)
    acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.3)
    instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.2)
    liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.1)
    valence = st.sidebar.slider("Valence", 0.0, 1.0, 0.5)
    tempo = st.sidebar.number_input("Tempo (BPM)", 0, 250, 120)
    spec_rate = st.sidebar.number_input("Spectral Rate", 0.0, 0.0001, 0.0, format="%.8f")

    data = {
        'duration (ms)': duration,
        'danceability': danceability,
        'energy': energy,
        'loudness': loudness,
        'speechiness': speechiness,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'valence': valence,
        'tempo': tempo,
        'spec_rate': spec_rate
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Emotion"):

    # Scale input features
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input).max()

    # Map numeric labels to emotions
    emotion_map = {0: "Sad", 1: "Happy", 2: "Energetic", 3: "Calm"}
    predicted_emotion = emotion_map[prediction]

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("Predicted Emotion")
    st.markdown(f"<h2 style='color:#1DB954'>{predicted_emotion}</h2>", unsafe_allow_html=True)
    st.write(f"Prediction confidence: **{prediction_proba*100:.2f}%**")

    st.subheader("Input Features")
    st.dataframe(input_df.T.rename(columns={0:"Value"}))

    # Probability bar chart
    proba_df = pd.DataFrame(model.predict_proba(scaled_input), columns=["Sad", "Happy", "Energetic", "Calm"])
    st.subheader("Prediction Probabilities")
    st.bar_chart(proba_df.T, width="stretch")
