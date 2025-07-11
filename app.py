import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import spacy

# Load spaCy and model artifacts
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
model = tf.keras.models.load_model("best_emotion_model (1).h5")
with open("vocab (4).pkl", "rb") as f:
    vocab = pickle.load(f)

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

MAX_LEN = 10

def preprocess_text(text, vocab, max_len=MAX_LEN):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop and not token.is_punct]
    ids = [vocab.get(token, 0) for token in tokens]
    ids = ids[:max_len] + [0]*(max_len - len(ids)) if len(ids) < max_len else ids[:max_len]
    return np.array([ids])

st.set_page_config(page_title="Emotion Detector", page_icon="😃", layout="centered")

st.markdown("""
    <style>
        body { font-family: 'Manrope', 'Noto Sans', sans-serif; }
        .big-title { font-size: 28px; font-weight: 800; margin-bottom: 10px; }
        .sub-title { font-size: 18px; color: #677583; }
        .emotion-box {
            display: flex; align-items: center; gap: 16px;
            background-color: #f1f2f4; padding: 12px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🎯 Emotion Detector</div>', unsafe_allow_html=True)
text_input = st.text_area("Enter text here", height=150, placeholder="Type something emotional...")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = preprocess_text(text_input, vocab)
        probs = model.predict(X)[0]
        top_idx = np.argmax(probs)
        emotion = emotion_labels[top_idx]
        confidence = round(probs[top_idx] * 100, 2)

        st.markdown(f"### Detected Emotion: `{emotion.title()}`")
        st.markdown(f"<p class='sub-title'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

# Examples
st.markdown("### 🔍 Examples")

examples = {
    "I'm so happy today!": "joy",
    "I feel down today.": "sadness",
    "This is so frustrating!": "anger"
}

for example_text, emotion in examples.items():
    with st.expander(example_text):
        st.markdown(f"<p class='sub-title'>Predicted Emotion: `{emotion.title()}`</p>", unsafe_allow_html=True)


