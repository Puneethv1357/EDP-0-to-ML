import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import altair as alt
import base64
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Model & Tokenizer ===
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

# === Set Background from Local Image ===
def set_background(image_path):
    abs_path = os.path.join(os.path.dirname(__file__), image_path)
    with open(abs_path, "rb") as f:
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

set_background("assets/bg_2.png")  # Make sure this matches your repo path!

# === Custom CSS Styling for Better Visibility ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: #ffffff;
}

.stApp {{
    color: white;
}}

textarea, .stTextArea textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    background-color: rgba(0, 0, 0, 0.4) !important;
    color: #fff !important;
    font-size: 16px !important;
    padding: 14px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4) !important;
}

textarea::placeholder {
    color: rgba(255, 255, 255, 0.5);
}

button[kind="primary"] {
    background-color: #ffffff22 !important;
    color: #ffffff !important;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
    border: 1px solid #ffffff55;
}

.title-block {
    text-align: center;
    margin-top: 10px;
    margin-bottom: 20px;
}

.title-block h1 {
    font-size: 34px;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 0 2px 6px rgba(0,0,0,0.6);
}

details summary {
    font-size: 16px;
    cursor: pointer;
    color: #f0f0f0;
}
details p {
    color: #cccccc;
}
</style>
""", unsafe_allow_html=True)

# === UI Title ===
st.markdown("""
<div class="title-block">
    <h1>Emotion Detector 💬</h1>
</div>
""", unsafe_allow_html=True)

# === Input ===
tweet = st.text_area(
    label="",
    placeholder="Type your message here...",
    height=140,
    label_visibility="collapsed"
)

# === Predict ===
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
        <div style='margin-top: 20px; display: flex; gap: 12px; align-items: center; color: #fff;'>
            <div style='background:rgba(255,255,255,0.1); border-radius:8px; width:44px; height:44px; display:flex; justify-content:center; align-items:center; font-size:22px;'>{emoji_map[predicted_label]}</div>
            <div>
                <p style='margin:0; font-size:18px; font-weight:600;'>{predicted_label.title()}</p>
                <p style='color: #ccc; font-size:14px; margin-top: 4px;'>Confidence: {confidence:.1%}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence Chart
        probs_df = pd.DataFrame({
            "Emotion": emotion_labels,
            "Confidence": prediction.flatten()
        })
        chart = alt.Chart(probs_df).mark_bar().encode(
            x=alt.X('Emotion', sort=None),
            y='Confidence',
            color=alt.value("white")
        ).properties(width=460)
        st.altair_chart(chart, use_container_width=True)

# === Examples ===
st.markdown("""
<hr>
<h4 style="color:#fff;">Examples</h4>
<details><summary>I’m so happy today!</summary><p>Joy</p></details>
<details><summary>I feel down today.</summary><p>Sadness</p></details>
<details><summary>This is so frustrating!</summary><p>Anger</p></details>
""", unsafe_allow_html=True)

