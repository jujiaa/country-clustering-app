import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =====================================================
# 1. KONFIGURASI
# =====================================================
st.set_page_config(
    page_title="Country Development Clustering",
    page_icon="🌍",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "country_clustering_model.joblib"

FEATURES = [
    "child_mort", "exports", "health", "imports", "income",
    "inflation", "life_expec", "total_fer", "gdpp"
]

# =====================================================
# 2. JUDUL APLIKASI
# =====================================================
st.title("🌍 Country Development Clustering")
st.write(
    "Aplikasi pengelompokan negara berdasarkan sembilan indikator "
    "sosial, ekonomi, dan kesehatan menggunakan algoritma K-Means."
)
st.divider()

# =====================================================
# 3. MEMUAT MODEL
# =====================================================
@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH.name}. "
            "Letakkan file model di folder yang sama dengan app.py."
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = {"transformer", "scaler", "model", "features", "metrics"}
    missing_keys = required_keys.difference(bundle.keys())
    if missing_keys:
        raise ValueError(
            f"Bundle model tidak memiliki komponen: {sorted(missing_keys)}"
        )

    saved_features = list(bundle["features"])
    if saved_features != FEATURES:
        raise ValueError(
            "Urutan/nama fitur pada bundle model tidak sama dengan fitur aplikasi. "
            f"Fitur model: {saved_features}"
        )

    return bundle


try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Model gagal dimuat: {exc}")
    st.stop()

transformer = bundle["transformer"]
scaler = bundle["scaler"]
model = bundle["model"]

st.success(
    f"Model berhasil dimuat | K-Means | "
    f"Jumlah cluster: {model.n_clusters}"
)

# =====================================================
# 4. TAB APLIKASI
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
        "Masukkan nama negara dan sembilan indikatornya. "
        "Nama negara akan ditampilkan bersama hasil prediksi."
    )

    with st.form("single_country_form"):
        country_name = st.text_input(
            "Nama Negara",
            placeholder="Contoh: Indonesia",
            help="Nama ini hanya sebagai identitas tampilan; model K-Means memprediksi cluster dari sembilan indikator."
        )

        st.markdown("**Indikator negara**")
        st.caption(
            "Gunakan satuan dan skala yang sama dengan dataset training. "
            "Nilai inflation boleh negatif."
        )

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
                    value="",
                    placeholder="Masukkan angka",
                    key=f"single_{feature}"
                )

        submitted = st.form_submit_button(
            "Prediksi Cluster",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            if not country_name.strip():
                raise ValueError("Nama negara wajib diisi.")

            empty_fields = [
                feature for feature in FEATURES
                if not raw_values[feature].strip()
            ]
            if empty_fields:
                raise ValueError(
                    "Lengkapi seluruh indikator. Belum diisi: "
                    + ", ".join(empty_fields)
                )

            values = {
                feature: float(raw_values[feature].strip())
                for feature in FEATURES
            }

            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError("Semua indikator harus berupa angka yang valid.")

            negative_features = [
                feature for feature, value in values.items()
                if feature != "inflation" and value < 0
            ]
            if negative_features:
                raise ValueError(
                    "Nilai tidak boleh negatif pada indikator: "
                    + ", ".join(negative_features)
                )

            input_data = pd.DataFrame(
                [[values[feature] for feature in FEATURES]],
                columns=FEATURES
            )

            # Gunakan preprocessing yang sama dengan proses training.
            transformed_data = transformer.transform(input_data)
            scaled_data = scaler.transform(transformed_data)
            predicted_cluster = int(model.predict(scaled_data)[0])

            st.divider()
            st.subheader("Hasil Prediksi")

            result_col1, result_col2 = st.columns(2)
            result_col1.metric("Negara", country_name.strip())
            result_col2.metric("Cluster Negara", f"Cluster {predicted_cluster}")

            st.success(
                f"Negara {country_name.strip()} berhasil dikelompokkan "
                f"ke dalam Cluster {predicted_cluster}."
            )

            st.info(
                "Nomor cluster merupakan label kategori, bukan peringkat "
                "kesejahteraan negara. Untuk menjelaskan apakah cluster "
                "termasuk kelompok negara dengan tingkat pembangunan tinggi "
                "atau rendah, gunakan profil/rata-rata indikator setiap cluster "
                "dari hasil analisis clustering."
            )

            result_table = pd.DataFrame({
                "Negara": [country_name.strip()],
                "Cluster": [predicted_cluster],
                **{feature: [values[feature]] for feature in FEATURES}
            })
            st.dataframe(result_table, use_container_width=True)

            result_csv = result_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh hasil prediksi negara",
                data=result_csv,
                file_name="hasil_prediksi_negara.csv",
                mime="text/csv"
            )

        except ValueError as exc:
            st.error(f"Input tidak valid: {exc}")
        except Exception as exc:
            st.error(f"Prediksi gagal diproses: {exc}")

# =====================================================
# 6. PREDIKSI BANYAK NEGARA DENGAN CSV
# =====================================================
with tab2:
    st.subheader("Prediksi Banyak Negara")
    st.write(
        "Unggah CSV yang berisi sembilan kolom indikator. "
        "Sertakan kolom `country` agar nama negara tampil pada hasil."
    )
    st.code(", ".join(["country"] + FEATURES))

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        key="country_csv"
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            missing_features = [
                feature for feature in FEATURES
                if feature not in input_df.columns
            ]

            if missing_features:
                st.error(
                    "Kolom indikator berikut tidak ditemukan: "
                    + ", ".join(missing_features)
                )
            else:
                model_input = input_df[FEATURES].copy()

                for col in FEATURES:
                    model_input[col] = pd.to_numeric(
                        model_input[col], errors="coerce"
                    )

                if model_input.isna().any().any():
                    bad_columns = model_input.columns[
                        model_input.isna().any()
                    ].tolist()
                    st.error(
                        "Ada nilai kosong atau non-numerik pada kolom: "
                        + ", ".join(bad_columns)
                    )
                elif not np.isfinite(
                    model_input.to_numpy(dtype=float)
                ).all():
                    st.error("CSV mengandung nilai tak hingga atau tidak valid.")
                else:
                    nonnegative_features = [
                        f for f in FEATURES if f != "inflation"
                    ]
                    negative_columns = [
                        f for f in nonnegative_features
                        if (model_input[f] < 0).any()
                    ]

                    if negative_columns:
                        st.error(
                            "Nilai negatif tidak diperbolehkan pada kolom: "
                            + ", ".join(negative_columns)
                        )
                    else:
                        transformed_data = transformer.transform(model_input)
                        scaled_data = scaler.transform(transformed_data)
                        predictions = model.predict(scaled_data).astype(int)

                        result_df = input_df.copy()
                        if "country" not in result_df.columns:
                            result_df.insert(
                                0,
                                "country",
                                [f"Negara/baris {i + 1}" for i in range(len(result_df))]
                            )
                        result_df["cluster"] = predictions

                        st.success(
                            f"Prediksi berhasil untuk {len(result_df)} data negara."
                        )
                        st.dataframe(result_df, use_container_width=True)

                        st.subheader("Distribusi Anggota Cluster")
                        st.bar_chart(
                            result_df["cluster"].value_counts().sort_index()
                        )

                        csv_data = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Unduh Hasil Clustering CSV",
                            data=csv_data,
                            file_name="hasil_clustering_negara.csv",
                            mime="text/csv"
                        )

        except Exception as exc:
            st.error(f"File CSV gagal diproses: {exc}")

# =====================================================
# 7. INFORMASI MODEL
# =====================================================
with tab3:
    st.subheader("Informasi dan Evaluasi Model")

    metrics = bundle["metrics"]
    metric_columns = st.columns(3)

    metric_definitions = [
        ("Silhouette Score", "silhouette"),
        ("Davies-Bouldin Index", "davies_bouldin"),
        ("Calinski-Harabasz Score", "calinski_harabasz")
    ]

    for col, (label, key) in zip(metric_columns, metric_definitions):
        if key in metrics:
            col.metric(label, f"{float(metrics[key]):.4f}")
        else:
            col.metric(label, "Tidak tersedia")

    st.divider()
    st.markdown("**Fitur yang digunakan**")
    st.write(FEATURES)

    st.markdown("**Tahapan preprocessing**")
    st.write(
        "PowerTransformer metode Yeo-Johnson, kemudian StandardScaler. "
        "Objek preprocessing yang digunakan adalah objek hasil training."
    )

    st.markdown("**Algoritma clustering**")
    st.write("K-Means")

    st.warning(
        "Cluster merupakan label hasil pengelompokan. Penamaan karakteristik "
        "cluster harus didasarkan pada tabel profil cluster dari tahap analisis."
    )

# =====================================================
# 8. FOOTER
# =====================================================
st.divider()
st.caption(
    "Country Development Clustering | Final Project Data Science | "
    "CRISP-DM - Deployment"
)
