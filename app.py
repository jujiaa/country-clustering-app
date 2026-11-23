
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =====================================================
# 1. KONFIGURASI APLIKASI
# =====================================================

st.set_page_config(
    page_title="Country Clustering",
    page_icon="🌍",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "country_clustering_model.joblib"

FEATURES = [
    "child_mort",
    "exports",
    "health",
    "imports",
    "income",
    "inflation",
    "life_expec",
    "total_fer",
    "gdpp"
]

# =====================================================
# 2. JUDUL APLIKASI
# =====================================================

st.title("🌍 Country Development Clustering")

st.write(
    "Aplikasi pengelompokan negara berdasarkan indikator "
    "sosial, ekonomi, dan kesehatan menggunakan algoritma K-Means."
)

st.markdown("---")

# =====================================================
# 3. MEMUAT MODEL MACHINE LEARNING
# =====================================================

@st.cache_resource
def load_bundle():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "File country_clustering_model.joblib tidak ditemukan. "
            "Pastikan file berada dalam folder yang sama dengan app.py."
        )

    bundle = joblib.load(MODEL_PATH)

    required = {
        "transformer",
        "scaler",
        "model",
        "features",
        "metrics"
    }

    missing = required.difference(bundle.keys())

    if missing:
        raise ValueError(
            f"Komponen model tidak lengkap: {sorted(missing)}"
        )

    if list(bundle["features"]) != FEATURES:
        raise ValueError(
            "Urutan fitur model tidak sesuai dengan aplikasi."
        )

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
    f"Model berhasil dimuat | "
    f"Algoritma: K-Means | "
    f"Jumlah cluster: {model.n_clusters}"
)

# =====================================================
# 4. MEMBUAT TAB APLIKASI
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "Prediksi Satu Negara",
    "Prediksi Banyak Negara (CSV)",
    "Informasi Model"
])

# =====================================================
# 5. PREDIKSI SATU NEGARA
# =====================================================

with tab1:

    st.subheader("Prediksi Cluster Negara")

    st.write(
        "Masukkan sembilan indikator negara untuk "
        "mengetahui kelompok cluster berdasarkan model K-Means."
    )

    st.caption(
        "Gunakan angka sesuai satuan dataset. "
        "Nilai inflation boleh negatif."
    )

    with st.form("single_country_form"):

        input_specs = [
            ("child_mort", "Child Mortality"),
            ("exports", "Exports (% GDP)"),
            ("health", "Health (% GDP)"),
            ("imports", "Imports (% GDP)"),
            ("income", "Income per Capita"),
            ("inflation", "Inflation (%)"),
            ("life_expec", "Life Expectancy"),
            ("total_fer", "Total Fertility"),
            ("gdpp", "GDP per Capita")
        ]

        columns = st.columns(3)

        raw_values = {}

        for i, (feature, label) in enumerate(input_specs):

            with columns[i % 3]:

                raw_values[feature] = st.text_input(
                    label,
                    value="0",
                    key=f"single_{feature}",
                    help="Masukkan angka. Gunakan titik untuk desimal."
                )

        submitted = st.form_submit_button(
            "Prediksi Cluster",
            type="primary"
        )

    if submitted:

        try:

            values = {
                feature: float(raw_values[feature].strip())
                for feature in FEATURES
            }

            if not all(
                np.isfinite(value)
                for value in values.values()
            ):
                raise ValueError(
                    "Semua input harus berupa angka yang valid."
                )

            negative_features = [
                feature
                for feature, value in values.items()
                if feature != "inflation" and value < 0
            ]

            if negative_features:

                raise ValueError(
                    "Nilai tidak boleh negatif pada fitur: "
                    + ", ".join(negative_features)
                )

            input_data = pd.DataFrame(
                [[values[f] for f in FEATURES]],
                columns=FEATURES
            )

            # Preprocessing menggunakan objek hasil training
            transformed = transformer.transform(input_data)

            scaled = scaler.transform(transformed)

            # Prediksi cluster
            prediction = int(
                model.predict(scaled)[0]
            )

            st.markdown("---")

            st.subheader("Hasil Prediksi")

            st.metric(
                label="Cluster Negara",
                value=f"Cluster {prediction}"
            )

            st.success(
                f"Data negara berhasil dikelompokkan "
                f"ke dalam Cluster {prediction}."
            )

            st.info(
                "Nomor cluster merupakan label kategori, "
                "bukan peringkat kesejahteraan negara. "
                "Interpretasi harus mengacu pada karakteristik "
                "rata-rata setiap cluster."
            )

        except ValueError as exc:

            st.error(f"Input tidak valid: {exc}")

        except Exception as exc:

            st.error(f"Prediksi gagal: {exc}")


# =====================================================
# 6. PREDIKSI BANYAK NEGARA MENGGUNAKAN CSV
# =====================================================

with tab2:

    st.subheader("Prediksi Banyak Negara")

    st.write(
        "Unggah file CSV yang berisi sembilan indikator "
        "negara untuk melakukan clustering secara bersamaan."
    )

    st.caption("Kolom fitur yang wajib tersedia:")

    st.code(", ".join(FEATURES))

    st.caption(
        "Kolom country boleh disertakan untuk mempertahankan "
        "identitas negara."
    )

    uploaded = st.file_uploader(
        "Upload Dataset CSV",
        type=["csv"]
    )

    if uploaded is not None:

        try:

            input_df = pd.read_csv(uploaded)

            missing = [
                feature
                for feature in FEATURES
                if feature not in input_df.columns
            ]

            if missing:

                st.error(
                    f"Kolom berikut belum tersedia: {missing}"
                )

            else:

                model_input = input_df[FEATURES].copy()

                for col in FEATURES:

                    model_input[col] = pd.to_numeric(
                        model_input[col],
                        errors="coerce"
                    )

                if model_input.isna().any().any():

                    bad_cols = model_input.columns[
                        model_input.isna().any()
                    ].tolist()

                    st.error(
                        "Terdapat nilai kosong atau non-numerik "
                        f"pada kolom: {bad_cols}"
                    )

                elif not np.isfinite(
                    model_input.to_numpy(dtype=float)
                ).all():

                    st.error(
                        "CSV mengandung nilai yang tidak valid."
                    )

                else:

                    nonnegative_features = [
                        f for f in FEATURES
                        if f != "inflation"
                    ]

                    negative_cols = [
                        f for f in nonnegative_features
                        if (model_input[f] < 0).any()
                    ]

                    if negative_cols:

                        st.error(
                            "Nilai negatif tidak diperbolehkan "
                            f"pada kolom: {negative_cols}"
                        )

                    else:

                        # Preprocessing data CSV
                        transformed = transformer.transform(
                            model_input
                        )

                        scaled = scaler.transform(
                            transformed
                        )

                        # Prediksi seluruh data
                        predictions = model.predict(
                            scaled
                        ).astype(int)

                        result = input_df.copy()

                        result["cluster"] = predictions

                        st.success(
                            f"Prediksi berhasil untuk "
                            f"{len(result)} negara."
                        )

                        st.subheader("Hasil Clustering")

                        st.dataframe(
                            result,
                            use_container_width=True
                        )

                        # Menghitung jumlah anggota setiap cluster
                        st.subheader(
                            "Distribusi Anggota Cluster"
                        )

                        cluster_counts = (
                            result["cluster"]
                            .value_counts()
                            .sort_index()
                        )

                        st.bar_chart(cluster_counts)

                        # Download hasil
                        csv_bytes = result.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Hasil Clustering",
                            data=csv_bytes,
                            file_name="hasil_clustering_negara.csv",
                            mime="text/csv"
                        )

        except Exception as exc:

            st.error(
                f"File CSV gagal diproses: {exc}"
            )


# =====================================================
# 7. INFORMASI DAN EVALUASI MODEL
# =====================================================

with tab3:

    st.subheader("Informasi Model Clustering")

    st.write(
        "Model yang digunakan adalah K-Means dengan "
        "preprocessing PowerTransformer dan StandardScaler."
    )

    metrics = bundle["metrics"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Silhouette Score",
        f"{metrics['silhouette']:.4f}"
    )

    col2.metric(
        "Davies-Bouldin Index",
        f"{metrics['davies_bouldin']:.4f}"
    )

    col3.metric(
        "Calinski-Harabasz Score",
        f"{metrics['calinski_harabasz']:.4f}"
    )

    st.markdown("---")

    st.subheader("Variabel yang Digunakan")

    st.write(FEATURES)

    st.subheader("Metode Preprocessing")

    st.write("""
    1. PowerTransformer menggunakan metode Yeo-Johnson.
    2. StandardScaler untuk menyamakan skala fitur.
    3. K-Means untuk melakukan pengelompokan negara.
    """)

    st.subheader("Algoritma")

    st.write("K-Means Clustering")

    st.markdown("---")

    st.warning(
        "Hasil clustering merupakan alat bantu analisis "
        "untuk mengidentifikasi kelompok negara dengan "
        "karakteristik yang relatif serupa. "
        "Nomor cluster bukan peringkat negara."
    )


# =====================================================
# 8. FOOTER APLIKASI
# =====================================================

st.markdown("---")

st.caption(
    "Country Development Clustering | "
    "Final Project Data Science | "
    "CRISP-DM - Deployment"
)
