import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import altair as alt
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
emoji_map = {'joy': '😊', 'sadness': '😢', 'anger': '😠', 'love': '❤️', 'fear': '😨', 'surprise': '😲'}
maxlen = 50

# Tailwind + Font styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Noto+Sans:wght@400;500;700;900&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Manrope', 'Noto Sans', sans-serif;
        background-color: #ffffff;
    }
    .title-section {
        padding: 20px 16px 8px 16px;
        background: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .textarea-box textarea {
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #dde0e4;
        font-size: 16px;
        resize: none;
        min-height: 140px;
        color: #121417;
    }
    .analyze-button {
        background-color: #327fcc;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App Title Section
st.markdown("""
<div class="title-section">
    <div>
        <svg width="24" height="24" fill="currentColor" viewBox="0 0 256 256">
            <path d="M224,128a8,8,0,0,1-8,8H59.31l58.35,58.34a8,8,0,0,1-11.32,11.32l-72-72a8,8,0,0,1,0-11.32l72-72a8,8,0,0,1,11.32,11.32L59.31,120H216A8,8,0,0,1,224,128Z"/>
        </svg>
    </div>
    <h2 style="font-size: 20px; font-weight: bold; margin: 0 auto;">Emotion Detector</h2>
</div>
""", unsafe_allow_html=True)

# Text Input Area
tweet = st.text_area(
    label="",
    placeholder="Enter text here...",
    height=140,
    label_visibility="collapsed"
)

# Predict Button
if st.button("Analyze", use_container_width=True):
    if not tweet.strip():
        st.warning("Please enter some text.")
    else:
        # Preprocess and predict
        seq = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        prediction = model.predict(padded)
        predicted_label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Result
        st.markdown(f"""
        <div style="padding: 20px 16px;">
            <h3 style="font-size: 18px; font-weight: bold;">Detected Emotion</h3>
            <div style="display: flex; align-items: center; gap: 12px; margin-top: 8px;">
                <div style="width: 40px; height: 40px; background-color: #f1f2f4; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">{emoji_map[predicted_label]}</div>
                <p style="font-size: 16px; margin: 0;">{predicted_label.title()}</p>
            </div>
            <p style="color: #677583; font-size: 14px; margin-top: 8px;">Confidence: {confidence:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

        # Chart
        probs_df = pd.DataFrame({
            "Emotion": emotion_labels,
            "Confidence": prediction.flatten()
        })
        chart = alt.Chart(probs_df).mark_bar().encode(
            x=alt.X('Emotion', sort=None),
            y='Confidence',
            color=alt.value("#327fcc")
        ).properties(width=500)
        st.altair_chart(chart, use_container_width=True)

# Example Section
st.markdown("""
<h3 style="padding: 16px; font-weight: bold;">Examples</h3>
<div style="padding: 0 16px 16px;">
<details style="margin-bottom: 10px;"><summary>I'm so happy today!</summary><p style="color: #677583;">Joy</p></details>
<details style="margin-bottom: 10px;"><summary>I feel down today.</summary><p style="color: #677583;">Sadness</p></details>
<details><summary>This is so frustrating!</summary><p style="color: #677583;">Anger</p></details>
</div>
""", unsafe_allow_html=True)
