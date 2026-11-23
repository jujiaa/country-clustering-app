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
            "Jalankan cell export pada notebook dan letakkan file tersebut "
            "di folder yang sama dengan app.py."
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
    st.caption("Isi nilai sesuai satuan pada dataset Country Data.")
    with st.form("single_country_form"):
        c1, c2, c3 = st.columns(3)
        values = {}
        input_specs = [
            ("child_mort", "Child mortality", 0.0),
            ("exports", "Exports (% GDP)", 0.0),
            ("health", "Health (% GDP)", 0.0),
            ("imports", "Imports (% GDP)", 0.0),
            ("income", "Income per capita", 0.0),
            ("inflation", "Inflation (%)", 0.0),
            ("life_expec", "Life expectancy (years)", 0.0),
            ("total_fer", "Total fertility", 0.0),
            ("gdpp", "GDP per capita", 0.0),
        ]
        columns = [c1, c2, c3]
        for i, (feature, label, default) in enumerate(input_specs):
            with columns[i % 3]:
                values[feature] = st.number_input(
                    label, min_value=0.0, value=float(default),
                    step=0.1, format="%.3f", key=f"single_{feature}"
                )
        submitted = st.form_submit_button("Prediksi cluster", type="primary")

    if submitted:
        row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        transformed = transformer.transform(row)
        scaled = scaler.transform(transformed)
        prediction = int(model.predict(scaled)[0])
        st.metric("Hasil prediksi", f"Cluster {prediction}")
        st.info(
            "Nomor cluster adalah label matematis, bukan peringkat negara. "
            "Interpretasi karakteristik cluster perlu dilihat dari profil "
            "rata-rata setiap cluster."
        )

with tab2:
    st.write("Isi tab 2")
