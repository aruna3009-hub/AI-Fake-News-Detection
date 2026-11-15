
import streamlit as st
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #E8EDF5;
    color: #142B4A;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main title */
.main-title {
    text-align: center;
    color: #142B4A;
    font-size: 38px;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #168C8C;
    font-size: 17px;
}

/* Section headings */
h1, h2, h3 {
    color: #142B4A !important;
}

/* Navigation */
.stRadio > div {
    background-color: white;
    border: 1px solid #2463A6;
    border-radius: 10px;
    padding: 10px;
}

/* Text area */
.stTextArea textarea {
    background-color: white !important;
    color: #142B4A !important;
    border: 1px solid #2463A6 !important;
    border-radius: 10px !important;
}

/* Dropdown */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: white;
    border-color: #168C8C;
    color: #142B4A;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background-color: #2463A6;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #168C8C;
    color: white;
    border: none;
}

/* Information cards */
.info-card {
    background-color: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #168C8C;
    margin-bottom: 15px;
}

/* Success and warning messages */
.stAlert[data-testid="stAlert"] {
    border-radius: 10px;
}

div[data-testid="stAlert"][kind="warning"] {
    border-left: 5px solid #D99127;
}
/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #168C8C;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #2463A6;
}

/* Footer */
.footer {
    text-align: center;
    color: #142B4A;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)
   
model_path = "models/fake_news_model.pkl"

if not os.path.exists(model_path):
    st.error("Model not found. Please check the models folder.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    '<div class="main-title">📰 AI-Based Fake News Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An intelligent text classification system '
    'using Machine Learning</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

menu = st.radio(
    "Navigation",
    ["Home", "Analysis History", "About Project"],
    horizontal=True
)

if menu == "Home":

    st.markdown("## 🔍 News Analysis")

    st.write(
        "Enter a news statement below. The trained machine learning "
        "model will analyse its text patterns and classify it."
    )

    sample_news = [
        "The government announced a new education policy.",
        "Scientists discovered a new planet.",
        "The president announced a major economic reform."
    ]

    st.markdown("### Try a sample statement")

    selected_sample = st.selectbox(
        "Choose a sample",
        ["Select a sample"] + sample_news
    )

    news_text = st.text_area(
        "Enter News Statement",
        value="" if selected_sample == "Select a sample"
        else selected_sample,
        height=170,
        placeholder="Type or paste your news statement here..."
    )

    st.caption(f"Characters entered: {len(news_text)}")

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

            st.divider()
            st.markdown("## 📊 Analysis Result")

            if prediction == "Reliable":
                st.success("Prediction: Reliable")
            else:
                st.warning("Prediction: Potentially Misleading")

            st.write("Model confidence score:")
            st.progress(float(confidence / 100))
            st.write(f"**{confidence:.2f}%**")

            st.session_state.history.insert(
                0,
                {
                    "news": news_text,
                    "prediction": prediction,
                    "confidence": confidence,
                    "time": datetime.now().strftime("%d-%m-%Y %I:%M %p")
                }
            )

            st.info(
                "This is a machine learning prediction based on "
                "patterns learned from the training dataset. "
                "It does not establish whether the news is true "
                "or false. Always verify information using "
                "reliable sources."
            )

elif menu == "Analysis History":

    st.markdown("## 📚 Analysis History")

    if not st.session_state.history:
        st.info("No analysis has been performed in this session yet.")

    else:
        st.write("Your recent news analysis results:")

        for index, item in enumerate(st.session_state.history, 1):

            with st.expander(f"Analysis {index} - {item['time']}"):

                st.write("**News Statement:**")
                st.write(item["news"])

                st.write("**Prediction:**", item["prediction"])

                st.write(
                    f"**Confidence:** {item['confidence']:.2f}%"
                )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

elif menu == "About Project":

    st.markdown("## 📘 About the Project")

    st.write(
        "AI-Based Fake News Detection is a Class XII project "
        "developed using Python and Machine Learning."
    )

    st.markdown("### Technologies Used")

    st.markdown("""
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorization
    - Logistic Regression
    - Streamlit
    - Joblib
    """)

    st.markdown("### How the System Works")

    st.markdown("""
    1. News statements are provided as input.
    2. TF-IDF converts text into numerical features.
    3. Logistic Regression analyses the extracted features.
    4. The model predicts a classification.
    5. The application displays the prediction and confidence.
    """)

    st.markdown("### Model Performance")

    st.metric("Test Accuracy", "61.33%")

    st.markdown("### Limitations")

    st.write(
        "The system identifies patterns in text rather than "
        "verifying claims against live news sources. It may "
        "produce incorrect predictions. The confidence score "
        "is not a measure of factual truth."
    )

    st.markdown("### Future Enhancements")

    st.write(
        "Future versions may include larger datasets, improved "
        "classification models, multilingual support and "
        "integration with trusted fact-checking sources."
    )

st.divider()

st.markdown(
    '<div class="footer">AI-Based Fake News Detection | '
    'Class XII Project</div>',
    unsafe_allow_html=True
)
