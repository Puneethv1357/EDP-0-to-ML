import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import altair as alt
import base64
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_emotion_model (1).keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer (4).pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
emoji_map = {
    'joy': '😊', 'sadness': '😢', 'anger': '😠',
    'love': '❤️', 'fear': '😨', 'surprise': '😲'
}
maxlen = 50

# ---------- Streamlit Cloud-Compatible Background ----------
def set_background_from_local(image_path):
    file_path = os.path.join(os.path.dirname(__file__), image_path)
    with open(file_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 👇 this path must match your folder structure
set_background_from_local("assets/anime_bg.png")

# ---------- CSS Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

textarea, .stTextArea textarea {
    border-radius: 12px !important;
    border: 1px solid #cccccc !important;
    background-color: rgba(255, 255, 255, 0.95);
    color: #333;
    font-size: 16px !important;
    padding: 14px !important;
    outline: none !important;
    box-shadow: none !important;
}

button[kind="secondary"], button[kind="primary"] {
    background-color: #327fcc !important;
    color: white !important;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
}

.title-block {
    text-align: center;
    margin-top: 10px;
    margin-bottom: 20px;
}

.title-block h1 {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.6);
}

details summary {
    font-size: 16px;
    cursor: pointer;
    color: #fff;
}

details p {
    color: #ccc;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("""
<div class="title-block">
    <h1>Emotion Detector 💬</h1>
</div>
""", unsafe_allow_html=True)

# ---------- Input ----------
tweet = st.text_area(
    label="",
    placeholder="Enter your text here...",
    height=140,
    label_visibility="collapsed"
)

# ---------- Predict ----------
if st.button("Analyze"):
    if not tweet.strip():
        st.error("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        prediction = model.predict(padded)
        predicted_label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown(f"""
        <div style='margin-top: 20px; display: flex; gap: 12px; align-items: center;'>
            <div style='background:#f1f2f4; border-radius:8px; width:44px; height:44px; display:flex; justify-content:center; align-items:center; font-size:22px;'>{emoji_map[predicted_label]}</div>
            <div>
                <p style='margin:0; font-size:18px; font-weight:600;'>{predicted_label.title()}</p>
                <p style='color: #677583; font-size:14px; margin-top: 4px;'>Confidence: {confidence:.1%}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Bar Chart
        probs_df = pd.DataFrame({
            "Emotion": emotion_labels,
            "Confidence": prediction.flatten()
        })
        chart = alt.Chart(probs_df).mark_bar().encode(
            x=alt.X('Emotion', sort=None),
            y='Confidence',
            color=alt.value("#327fcc")
        ).properties(width=460)
        st.altair_chart(chart, use_container_width=True)

# ---------- Examples ----------
st.markdown("""
<hr>
<h4 style="color:#fff;">Examples</h4>
<details><summary>I’m so happy today!</summary><p>Joy</p></details>
<details><summary>I feel down today.</summary><p>Sadness</p></details>
<details><summary>This is so frustrating!</summary><p>Anger</p></details>
""", unsafe_allow_html=True)

