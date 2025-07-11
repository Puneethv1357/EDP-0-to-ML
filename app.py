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

# ---------- HTML + CSS ----------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap" rel="stylesheet">
<style>
body {
    font-family: 'Manrope', sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
}
.container {
    max-width: 480px;
    margin: 0 auto;
    padding: 24px 16px;
    background: var(--background-color);
    border-radius: 12px;
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
.custom-warning {
    color: #ffa726;
    background-color: #fff3e0;
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Layout Container ----------
st.markdown('<div class="container">', unsafe_allow_html=True)

# ---------- Text Input ----------
tweet = st.text_area(
    label="",
    placeholder="Enter your text here...",
    height=140,
    label_visibility="collapsed"
)

# ---------- Button + Logic ----------
if st.button("Analyze"):
    if not tweet.strip():
        st.markdown('<div class="custom-warning">Please enter some text to analyze.</div>', unsafe_allow_html=True)
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        prediction = model.predict(padded)
        predicted_label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Result
        st.markdown(f"""
        <div style='margin-top: 20px; display: flex; gap: 12px; align-items: center;'>
            <div style='background:#f1f2f4; border-radius:8px; width:44px; height:44px; display:flex; justify-content:center; align-items:center; font-size:22px;'>{emoji_map[predicted_label]}</div>
            <div>
                <p style='margin:0; font-size:18px; font-weight:600;'>{predicted_label.title()}</p>
                <p style='color: #677583; font-size:14px; margin-top: 4px;'>Confidence: {confidence:.1%}</p>
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
<details style="margin-bottom: 10px;"><summary>I’m so happy today!</summary><p style="color:#677583;">Joy</p></details>
<details style="margin-bottom: 10px;"><summary>I feel down today.</summary><p style="color:#677583;">Sadness</p></details>
<details style="margin-bottom: 10px;"><summary>This is so frustrating!</summary><p style="color:#677583;">Anger</p></details>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
