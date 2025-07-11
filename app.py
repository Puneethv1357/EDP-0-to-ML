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
emoji_map = {
    'joy': '😊', 'sadness': '😢', 'anger': '😠',
    'love': '❤️', 'fear': '😨', 'surprise': '😲'
}
maxlen = 50

# ---------- HTML + CSS Styling ----------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap" rel="stylesheet">
<style>
body {
    font-family: 'Manrope', sans-serif;
    background-color: #f9fafb;
}
.container {
    max-width: 480px;
    margin: 0 auto;
    padding: 24px 16px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(0,0,0,0.05);
}
.title-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}
.title-bar h2 {
    font-size: 22px;
    font-weight: 700;
    color: #121417;
    text-align: center;
    flex-grow: 1;
}
textarea {
    border-radius: 12px !important;
    border: 1px solid #dde0e4 !important;
    padding: 15px !important;
    font-size: 16px !important;
}
button {
    background-color: #327fcc;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    margin-top: 10px;
}
.result-box {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 24px;
}
.result-icon {
    background: #f1f2f4;
    border-radius: 8px;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}
.confidence {
    color: #677583;
    font-size: 14px;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("""
<div class="title-bar">
    <div style="width:24px;"></div>
    <h2>Emotion Detector</h2>
    <div style="width:24px;"></div>
</div>
""", unsafe_allow_html=True)

# ---------- Text Input ----------
tweet = st.text_area(
    label="",
    placeholder="Enter your text here...",
    height=140,
    label_visibility="collapsed"
)

# ---------- Predict Button ----------
if st.button("Analyze"):
    if not tweet.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        prediction = model.predict(padded)
        predicted_label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Result
        st.markdown(f"""
        <div class="result-box">
            <div class="result-icon">{emoji_map[predicted_label]}</div>
            <div>
                <p style="margin: 0; font-size: 18px; font-weight: 600;">{predicted_label.title()}</p>
                <p class="confidence">Confidence: {confidence:.1%}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---------- Chart ----------
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
<hr style="margin-top: 32px; margin-bottom: 16px;">
<h4 style="margin-bottom: 12px;">Examples</h4>
<details style="margin-bottom: 10px;"><summary>I’m so happy today!</summary><p class="confidence">Joy</p></details>
<details style="margin-bottom: 10px;"><summary>I feel down today.</summary><p class="confidence">Sadness</p></details>
<details style="margin-bottom: 10px;"><summary>This is so frustrating!</summary><p class="confidence">Anger</p></details>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close .container
