
import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
}
.main-title {
    text-align: center;
    color: #173b68;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #52677f;
    font-size: 16px;
}
.result-card {
    background-color: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #dce5f0;
    margin-top: 15px;
}
.footer {
    text-align: center;
    color: #718096;
    font-size: 13px;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📰 AI-Based Fake News Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyse news statements using Machine Learning</div>',
    unsafe_allow_html=True
)

st.write("")

st.markdown("### 🔍 News Analysis")

st.write(
    "Enter a news statement below to analyse its classification."
)

model_path = "models/fake_news_model.pkl"

if not os.path.exists(model_path):
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

with st.expander("View sample news statements"):
    st.write("1. The government announced a new education policy.")
    st.write("2. Scientists discovered a new planet.")
    st.write("3. The president announced a major economic reform.")

news_text = st.text_area(
    "Enter News Statement",
    height=150,
    placeholder="Type or paste your news statement here..."
)

analyse = st.button(
    "🔎 Analyse News",
    use_container_width=True
)

if analyse:
    if not news_text.strip():
        st.warning("Please enter a news statement first.")

    else:
        prediction = model.predict([news_text])[0]
        probabilities = model.predict_proba([news_text])[0]
        confidence = max(probabilities) * 100

        st.markdown("### 📊 Analysis Result")

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if prediction == "Reliable":
            st.success("Prediction: Reliable")
        else:
            st.warning("Prediction: Potentially Misleading")

        st.write("Model confidence score:")
        st.progress(float(confidence / 100))
        st.write(f"**{confidence:.2f}%**")

        st.markdown('</div>', unsafe_allow_html=True)

        st.info(
            "Important: This prediction is based on patterns "
            "learned from the training dataset. It does not "
            "confirm whether the statement is factually true. "
            "Always verify news using reliable sources."
        )

st.markdown(
    '<div class="footer">AI-Based Fake News Detection | Class XII Project</div>',
    unsafe_allow_html=True
)
