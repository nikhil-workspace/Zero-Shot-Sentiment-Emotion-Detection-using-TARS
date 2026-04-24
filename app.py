## Streamlit file for UI

# --- app_tars.py ---
import streamlit as st
from flair.models import TARSClassifier
from flair.data import Sentence
import torch

# --- Fix PyTorch >= 2.6 weights issue ---
orig_load = torch.load
def torch_load_fixed(*args, **kwargs):
    kwargs["weights_only"] = False
    return orig_load(*args, **kwargs)
torch.load = torch_load_fixed

# --- Page Setup ---
st.set_page_config(
    page_title="TARS Emotion Detector",
    page_icon="🎭",
    layout="centered",
)

# --- Custom CSS ---
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #fce7f3, #fae8ff, #dbeafe);
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
}

h1 {
    text-align: center;
    background: -webkit-linear-gradient(45deg, #ec4899, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5em;
}

p {
    text-align: center;
    color: #555;
}

.result-card {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 18px;
    padding: 25px 30px;
    box-shadow: 0 4px 25px rgba(0,0,0,0.1);
    margin-top: 25px;
    backdrop-filter: blur(8px);
}

.progress-container {
    background-color: #f1f5f9;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    margin-top: 4px;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #f97316, #ec4899);
    border-radius: 8px;
}

.emotion-label {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    color: #374151;
}
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
st.info("🔄 Loading TARS model (please wait a few seconds)...")
tars = TARSClassifier.load("tars-base")
st.success("✅ Model loaded successfully!")

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise"
]
sentiment_labels = ["positive", "negative", "neutral"]

# --- UI Header ---
st.markdown("<h1>🎭 Zero-Shot Emotion & Sentiment Detection</h1>", unsafe_allow_html=True)
st.markdown("<p>Analyze emotions and sentiment in your text using the TARS model</p>", unsafe_allow_html=True)

text_input = st.text_area("💬 Enter your text:", height=150, placeholder="Type or paste your text here...")

# --- Main Button ---
if st.button("🔍 Analyze"):
    if text_input.strip():
        with st.spinner("Analyzing emotions and sentiment..."):
            # Emotion Prediction
            tars.switch_to_task("zero-shot-emotion")
            sentence_emotion = Sentence(text_input)
            tars.predict_zero_shot(sentence_emotion, emotion_labels, multi_label=True)

            predicted_emotions = sorted(
                [(lbl.value, round(lbl.score * 100, 2)) for lbl in sentence_emotion.labels],
                key=lambda x: x[1],
                reverse=True
            )

            # Sentiment Prediction
            tars.switch_to_task("zero-shot-sentiment")
            sentence_sentiment = Sentence(text_input)
            tars.predict_zero_shot(sentence_sentiment, sentiment_labels)
            sentiment_result = sentence_sentiment.labels[0].value if sentence_sentiment.labels else "neutral"



        # Sentiment Section
        color_map = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#6b7280"}
        st.markdown("<h3> Sentiment:</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='text-align:center; font-size:22px; font-weight:bold; color:{color_map.get(sentiment_result)};'>"
            f"{sentiment_result.capitalize()}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3> Emotions:</h3>", unsafe_allow_html=True)

        for emotion, score in predicted_emotions[:5]:
            st.markdown(
                f"""
                <div class="emotion-label">
                    <span>{emotion.capitalize()}</span><span>{score}%</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width:{score}%;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text first!")

st.markdown("<br><center><small>Developed with ❤️ using Flair + Streamlit</small></center>", unsafe_allow_html=True)
