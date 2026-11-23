import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Country Clustering", page_icon="🌍", layout="wide")

MODEL_PATH = Path(__file__).parent / "country_clustering_model.joblib"

FEATURES = [
    "child_mort", "exports", "health", "imports", "income",
    "inflation", "life_expec", "total_fer", "gdpp"
]

st.title("🌍 Country Development Clustering")
st.write(
    "Aplikasi pengelompokan negara menggunakan model K-Means "
    "yang dilatih pada dataset Country Data."
)

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "File country_clustering_model.joblib tidak ditemukan. "
            "Letakkan file tersebut di folder yang sama dengan app.py."
        )
    bundle = joblib.load(MODEL_PATH)
    required = {"transformer", "scaler", "model", "features", "metrics"}
    missing = required.difference(bundle.keys())
    if missing:
        raise ValueError(f"Isi model bundle tidak lengkap: {sorted(missing)}")
    if list(bundle["features"]) != FEATURES:
        raise ValueError("Urutan fitur dalam model tidak sesuai dengan aplikasi.")
    return bundle

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Gagal memuat model: {exc}")
    st.stop()

transformer = bundle["transformer"]
scaler = bundle["scaler"]
model = bundle["model"]

st.success(
    f"Model berhasil dimuat | Algoritma: K-Means | "
    f"Jumlah cluster: {model.n_clusters}"
)

tab1, tab2, tab3 = st.tabs(
    ["Prediksi satu negara", "Prediksi banyak negara (CSV)", "Informasi model"]
)

with tab1:
    st.subheader("Masukkan indikator negara")
    st.caption("Masukkan angka sesuai satuan pada dataset Country Data.")
    st.caption("Catatan: nilai inflation dapat negatif; indikator lainnya harus nol atau positif.")

    with st.form("single_country_form"):
        input_specs = [
            ("child_mort", "Child mortality", "0.0"),
            ("exports", "Exports (% GDP)", "0.0"),
            ("health", "Health (% GDP)", "0.0"),
            ("imports", "Imports (% GDP)", "0.0"),
            ("income", "Income per capita", "0.0"),
            ("inflation", "Inflation (%)", "0.0"),
            ("life_expec", "Life expectancy (years)", "0.0"),
            ("total_fer", "Total fertility", "0.0"),
            ("gdpp", "GDP per capita", "0.0"),
        ]
        columns = st.columns(3)
        raw_values = {}
        for i, (feature, label, default) in enumerate(input_specs):
            with columns[i % 3]:
                raw_values[feature] = st.text_input(
                    label, value=default, key=f"single_{feature}",
                    help="Masukkan angka; gunakan titik untuk desimal."
                )
        submitted = st.form_submit_button("Prediksi cluster", type="primary")

    if submitted:
        try:
            values = {feature: float(raw_values[feature].strip()) for feature in FEATURES}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("Semua nilai harus berupa angka yang valid.")
            negative_features = [
                feature for feature, value in values.items()
                if feature != "inflation" and value < 0
            ]
            if negative_features:
                raise ValueError(
                    "Nilai tidak boleh negatif untuk fitur: "
                    + ", ".join(negative_features)
