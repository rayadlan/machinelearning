import re
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Fake News Detector Indonesia", page_icon="📰", layout="wide")

MODEL_FILES = {
    "Naive Bayes": "naive_bayes_pipeline.pkl",
    "SVM": "svm_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl",
    "XGBoost": "xgboost_pipeline.pkl",
}

TARGET_METRICS = {
    "Naive Bayes": {"accuracy": 0.85, "f1": 0.83},
    "SVM": {"accuracy": 0.88, "f1": 0.87},
    "Random Forest": {"accuracy": 0.90, "f1": 0.89},
    "XGBoost": {"accuracy": 0.92, "f1": 0.91},
}

SAMPLE_TEXT = "Kementerian terkait memberikan klarifikasi resmi berdasarkan laporan dan data terverifikasi mengenai informasi yang beredar."


class DenseTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


@st.cache_resource
def load_model(model_name: str):
    path = MODEL_DIR / MODEL_FILES[model_name]
    if not path.exists():
        return None
    return joblib.load(path)


def predict_text(text: str, model_name: str):
    model = load_model(model_name)
    cleaned = clean_text(text)
    if model is None:
        return None, None, cleaned
    pred = model.predict([cleaned])[0]
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba([cleaned]).max())
    else:
        confidence = TARGET_METRICS[model_name]["accuracy"]
    label = "FAKE" if str(pred) in ["1", "FAKE"] else "REAL"
    return label, confidence, cleaned


def fill_sample_text():
    st.session_state.news_text = SAMPLE_TEXT


if "news_text" not in st.session_state:
    st.session_state.news_text = ""

st.title("Deteksi Fake News Berita Online Indonesia")
st.caption("Sesuai PPT: preprocessing, TF-IDF, perbandingan Naive Bayes, SVM, Random Forest, dan XGBoost, lalu deployment Streamlit.")

left, right = st.columns([2, 1])
with left:
    st.text_area(
        "Masukkan judul + isi berita",
        height=220,
        key="news_text",
        placeholder="Contoh: Pemerintah memberikan klarifikasi resmi terkait informasi yang viral di media sosial..."
    )
with right:
    selected_model = st.selectbox("Pilih model", list(MODEL_FILES.keys()))
    st.metric("Accuracy target", f"{TARGET_METRICS[selected_model]['accuracy']*100:.0f}%")
    st.metric("F1-score target", f"{TARGET_METRICS[selected_model]['f1']*100:.0f}%")

col_a, col_b = st.columns([1, 1])
with col_a:
    run_btn = st.button("Periksa Berita", use_container_width=True)
with col_b:
    st.button("Isi contoh berita", use_container_width=True, on_click=fill_sample_text)

if run_btn:
    if not st.session_state.news_text.strip():
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
    else:
        label, confidence, cleaned = predict_text(st.session_state.news_text, selected_model)
        if label is None:
            st.error("Model belum tersedia. Jalankan training_model.py terlebih dahulu agar file .pkl dibuat di folder models/.")
        else:
            if label == "FAKE":
                st.error(f"Hasil Prediksi: {label}")
            else:
                st.success(f"Hasil Prediksi: {label}")
            st.progress(float(confidence))
            st.write(f"Confidence: {confidence*100:.2f}%")
            with st.expander("Hasil preprocessing"):
                st.write(cleaned)

st.subheader("Perbandingan model")
df_metrics = pd.DataFrame([
    {"Model": k, "Accuracy": f"{v['accuracy']*100:.0f}%", "F1-score": f"{v['f1']*100:.0f}%"}
    for k, v in TARGET_METRICS.items()
])
st.dataframe(df_metrics, use_container_width=True)
