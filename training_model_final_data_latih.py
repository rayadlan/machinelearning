from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


class DenseTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_dataset() -> pd.DataFrame:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("Tidak ada file CSV di folder data/. Tambahkan dataset terlebih dahulu.")

    target_file = None
    for f in csv_files:
        if f.name.lower() == "Data_latih_50_50_oversampling.csv":
            target_file = f
            break
    if target_file is None:
        target_file = csv_files[0]

    df = pd.read_csv(target_file)
    cols = {c.lower(): c for c in df.columns}

    label_col = None
    for candidate in ["label", "class", "target", "category"]:
        if candidate in cols:
            label_col = cols[candidate]
            break

    judul_col = cols.get("judul")
    narasi_col = cols.get("narasi")
    title_col = cols.get("title")
    text_col = None
    for candidate in ["text", "content", "news", "article", "body", "narasi"]:
        if candidate in cols:
            text_col = cols[candidate]
            break

    if judul_col and narasi_col:
        df["combined_text"] = df[judul_col].astype(str) + " " + df[narasi_col].astype(str)
        text_col = "combined_text"
    elif title_col and text_col is not None:
        df["combined_text"] = df[title_col].astype(str) + " " + df[text_col].astype(str)
        text_col = "combined_text"
    elif judul_col and text_col is not None and text_col != judul_col:
        df["combined_text"] = df[judul_col].astype(str) + " " + df[text_col].astype(str)
        text_col = "combined_text"
    elif text_col is None and judul_col:
        text_col = judul_col
    elif text_col is None and title_col:
        text_col = title_col

    if text_col is None or label_col is None:
        raise ValueError("Dataset harus memiliki kolom teks dan label. Untuk dataset ini gunakan kolom judul, narasi, dan label.")

    df = df[[text_col, label_col]].dropna().copy()
    df.columns = ["text", "label"]
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(str).str.strip().str.upper().replace({
        "1": "FAKE",
        "0": "REAL",
        "HOAX": "FAKE",
        "VALID": "REAL"
    })
    df = df[df["label"].isin(["FAKE", "REAL"])]

    if df.empty:
        raise ValueError("Dataset kosong setelah proses pembersihan label. Cek isi kolom label pada Data_latih.csv.")

    print(f"Dataset digunakan: {target_file.name}")
    print(f"Jumlah data valid: {len(df)}")
    print("Distribusi label:")
    print(df["label"].value_counts())
    return df


def build_models():
    return {
        "naive_bayes_pipeline.pkl": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", MultinomialNB())
        ]),
        "svm_pipeline.pkl": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1, 2))),
            ("clf", SVC(kernel="linear", probability=True, random_state=42))
        ]),
        "random_forest_pipeline.pkl": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("to_dense", DenseTransformer()),
            ("clf", RandomForestClassifier(n_estimators=250, random_state=42))
        ]),
        "xgboost_pipeline.pkl": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("to_dense", DenseTransformer()),
            ("clf", XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42
            ))
        ])
    }


if __name__ == "__main__":
    df = load_dataset()
    df["y"] = df["label"].map({"REAL": 0, "FAKE": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["y"],
        test_size=0.2,
        random_state=42,
        stratify=df["y"]
    )

    models = build_models()
    rows = []

    for filename, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_label = np.where(preds == 1, "FAKE", "REAL")
        y_test_label = np.where(y_test == 1, "FAKE", "REAL")

        acc = accuracy_score(y_test_label, preds_label)
        precision = precision_score(y_test_label, preds_label, pos_label="FAKE")
        recall = recall_score(y_test_label, preds_label, pos_label="FAKE")
        f1 = f1_score(y_test_label, preds_label, pos_label="FAKE")

        rows.append({
            "model_file": filename,
            "accuracy": round(acc, 4),
            "precision_fake": round(precision, 4),
            "recall_fake": round(recall, 4),
            "f1_fake": round(f1, 4)
        })

        joblib.dump(model, MODEL_DIR / filename)
        print("=" * 60)
        print(filename)
        print(classification_report(y_test_label, preds_label))

    pd.DataFrame(rows).to_csv(BASE_DIR / "evaluation_results.csv", index=False)
    print("Hasil evaluasi disimpan ke evaluation_results.csv")
