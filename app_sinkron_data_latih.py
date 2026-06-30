import re
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
EVAL_FILE = BASE_DIR / "evaluation_results.csv"

st.set_page_config(page_title="Fake News Detector Indonesia", page_icon="📰", layout="wide")

MODEL_FILES = {
    "Naive Bayes": "naive_bayes_pipeline.pkl",
    "SVM": "svm_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl",
    "XGBoost": "xgboost_pipeline.pkl",
}

SAMPLE_TEXT = "Universitas dan pihak kampus memberikan klarifikasi resmi bahwa kegiatan akademik tetap berjalan normal berdasarkan informasi terverifikasi."


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
        confidence = None
    label = "FAKE" if str(pred) in ["1", "FAKE"] else "REAL"
    return label, confidence, cleaned


def load_evaluation_results():
    if EVAL_FILE.exists():
        df = pd.read_csv(EVAL_FILE)
        df["model_name"] = df["model_file"].replace({
            "naive_bayes_pipeline.pkl": "Naive Bayes",
            "svm_pipeline.pkl": "SVM",
            "random_forest_pipeline.pkl": "Random Forest",
            "xgboost_pipeline.pkl": "XGBoost",
        })
        return df
    return None


def fill_sample_text():
    st.session_state.news_text = SAMPLE_TEXT


if "news_text" not in st.session_state:
    st.session_state.news_text = ""

st.title("Deteksi Fake News Berita Online Indonesia")
st.caption("Menggunakan dataset Data_latih.csv dengan fitur judul + narasi, TF-IDF, dan model Naive Bayes, SVM, Random Forest, serta XGBoost.")

eval_df = load_evaluation_results()
if eval_df is not None and not eval_df.empty:
    best_row = eval_df.sort_values(by="accuracy", ascending=False).iloc[0]
    st.info(f"Model terbaik saat ini: {best_row['model_name']} | Accuracy: {best_row['accuracy']:.4f} | F1-score: {best_row['f1_fake']:.4f}")
else:
    st.warning("evaluation_results.csv belum tersedia. Jalankan training_model_final_data_latih.py terlebih dahulu.")

left, right = st.columns([2, 1])
with left:
    st.text_area(
        "Masukkan judul + isi/narasi berita",
        height=220,
        key="news_text",
        placeholder="Contoh: Universitas dan pihak kampus memberikan klarifikasi resmi bahwa kegiatan akademik tetap berjalan normal..."
    )
with right:
    selected_model = st.selectbox("Pilih model", list(MODEL_FILES.keys()))
    if eval_df is not None and not eval_df.empty:
        selected_row = eval_df[eval_df["model_name"] == selected_model]
        if not selected_row.empty:
            row = selected_row.iloc[0]
            st.metric("Accuracy", f"{row['accuracy']*100:.2f}%")
            st.metric("F1-score", f"{row['f1_fake']*100:.2f}%")

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
            st.error("Model belum tersedia. Jalankan training_model_final_data_latih.py terlebih dahulu agar file .pkl dibuat di folder models/.")
        else:
            if label == "FAKE":
                st.error(f"Hasil Prediksi: {label}")
            else:
                st.success(f"Hasil Prediksi: {label}")

            if confidence is not None:
                st.progress(float(confidence))
                st.write(f"Confidence: {confidence*100:.2f}%")
            else:
                st.write("Confidence tidak tersedia untuk model ini.")

            with st.expander("Hasil preprocessing"):
                st.write(cleaned)

if eval_df is not None and not eval_df.empty:
    st.subheader("Hasil evaluasi model")
    display_df = eval_df[["model_name", "accuracy", "precision_fake", "recall_fake", "f1_fake"]].copy()
    display_df.columns = ["Model", "Accuracy", "Precision (Fake)", "Recall (Fake)", "F1-score (Fake)"]
    st.dataframe(display_df, use_container_width=True)
