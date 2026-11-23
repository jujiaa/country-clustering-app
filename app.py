import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# 1. KONFIGURASI APLIKASI
# ============================================================
st.set_page_config(
    page_title="Country Development Clustering",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "country_clustering_model.joblib"

FEATURES = [
    "child_mort",
    "exports",
    "health",
    "imports",
    "income",
    "inflation",
    "life_expec",
    "total_fer",
    "gdpp",
]

FEATURE_INFO = {
    "child_mort": ("Child mortality", "Kematian anak per 1.000 kelahiran hidup"),
    "exports": ("Exports", "Ekspor barang dan jasa (% GDP)"),
    "health": ("Health", "Belanja kesehatan (% GDP)"),
    "imports": ("Imports", "Impor barang dan jasa (% GDP)"),
    "income": ("Income", "Pendapatan bersih per orang (USD)"),
    "inflation": ("Inflation", "Inflasi tahunan (%) — boleh bernilai negatif"),
    "life_expec": ("Life expectancy", "Harapan hidup saat lahir (tahun)"),
    "total_fer": ("Total fertility", "Jumlah anak per perempuan"),
    "gdpp": ("GDP per capita", "GDP per kapita (USD)"),
}

st.title("🌍 Country Development Clustering")
st.write(
    "Aplikasi pengelompokan negara berdasarkan sembilan indikator "
    "sosial, ekonomi, dan kesehatan menggunakan algoritma K-Means."
)

# ============================================================
# 2. MEMUAT MODEL DAN PREPROCESSOR
# ============================================================
@st.cache_resource(show_spinner="Memuat model clustering...")
def load_model_bundle(path_string):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {path.name}. "
            "Pastikan file country_clustering_model.joblib berada satu folder dengan app.py."
        )

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise TypeError(
            "Format file model bukan dictionary/bundle yang diharapkan. "
            "Periksa kembali cell export model pada notebook."
        )

    def find_object(possible_names, object_label):
        for key in possible_names:
            if key in bundle and bundle[key] is not None:
                return bundle[key]
        raise KeyError(
            f"Komponen '{object_label}' tidak ditemukan dalam model bundle. "
            f"Key yang tersedia: {list(bundle.keys())}"
        )

    transformer = find_object(
        ["transformer", "power_transformer", "pt", "powertransformer"],
        "PowerTransformer"
    )
    scaler = find_object(
        ["scaler", "standard_scaler", "std_scaler"],
        "StandardScaler"
    )
    model = find_object(
        ["model", "kmeans", "kmeans_final", "final_model", "best_kmeans"],
        "model K-Means"
    )

    stored_features = bundle.get("features", FEATURES)
    if list(stored_features) != FEATURES:
        raise ValueError(
            "Urutan/nama fitur dalam bundle berbeda dari aplikasi. "
            f"Fitur bundle: {list(stored_features)}"
        )

    return {
        "transformer": transformer,
        "scaler": scaler,
        "model": model,
        "metrics": bundle.get("metrics", {}),
        "features": list(stored_features),
        "raw_bundle": bundle,
    }


try:
    components = load_model_bundle(str(MODEL_PATH))
    transformer = components["transformer"]
    scaler = components["scaler"]
    model = components["model"]
    metrics = components["metrics"]

    cluster_count = getattr(model, "n_clusters", "tidak diketahui")
    st.success(
        f"Model berhasil dimuat | Algoritma: K-Means | "
        f"Jumlah cluster: {cluster_count}"
    )
except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.info(
        "Pastikan nama file model benar dan struktur bundle sesuai dengan "
        "objek PowerTransformer, StandardScaler, dan K-Means dari notebook."
    )
    st.stop()


# ============================================================
# 3. FUNGSI PREDIKSI DAN VALIDASI
# ============================================================
def validate_numeric_dataframe(dataframe):
    """Memastikan semua fitur tersedia, numerik, dan tidak kosong."""
    missing_columns = [col for col in FEATURES if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "Kolom berikut belum tersedia: " + ", ".join(missing_columns)
        )

    clean = dataframe[FEATURES].copy()

    for column in FEATURES:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    if clean.isna().any().any():
        invalid_columns = clean.columns[clean.isna().any()].tolist()
        raise ValueError(
            "Ada nilai kosong atau bukan angka pada kolom: "
            + ", ".join(invalid_columns)
        )

    values = clean.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Data mengandung nilai tak hingga atau tidak valid.")

    # Dalam dataset Country Data, inflasi dapat bernilai negatif.
    # Delapan indikator lainnya diharapkan tidak negatif.
    non_negative_features = [col for col in FEATURES if col != "inflation"]
    negative_columns = [
        col for col in non_negative_features if (clean[col] < 0).any()
    ]
    if negative_columns:
        raise ValueError(
            "Nilai negatif tidak diperbolehkan untuk: "
            + ", ".join(negative_columns)
            + ". Nilai inflation boleh negatif."
        )

    return clean


def predict_clusters(dataframe):
    clean = validate_numeric_dataframe(dataframe)
    transformed = transformer.transform(clean)
    scaled = scaler.transform(transformed)
    predictions = model.predict(scaled)
    return predictions.astype(int)


def parse_form_value(raw_value, feature):
    value_text = raw_value.strip()
    if not value_text:
        raise ValueError(f"Indikator {feature} belum diisi.")

    # Input manual menerima titik atau koma sebagai pemisah desimal.
    value_text = value_text.replace(",", ".")
    try:
        value = float(value_text)
    except ValueError:
        raise ValueError(f"Nilai {feature} harus berupa angka.")

    if not np.isfinite(value):
        raise ValueError(f"Nilai {feature} harus berupa angka yang valid.")

    if feature != "inflation" and value < 0:
        raise ValueError(f"Nilai {feature} tidak boleh negatif.")

    return value


# ============================================================
# 4. ANTARMUKA APLIKASI
# ============================================================
tab_single, tab_csv, tab_info = st.tabs(
    ["Prediksi satu negara", "Prediksi banyak negara (CSV)", "Informasi model"]
)

with tab_single:
    st.subheader("Prediksi cluster satu negara")
    st.write(
        "Isi nama negara dan seluruh indikator berdasarkan satuan dataset "
        "Country Data. Semua indikator wajib diisi."
    )

    with st.form("single_country_form", clear_on_submit=False):
        country_name = st.text_input(
            "Nama negara",
            placeholder="Contoh: Indonesia"
        )

        st.markdown("**Indikator sosial, ekonomi, dan kesehatan**")
        columns = st.columns(3)
        form_values = {}

        for index, feature in enumerate(FEATURES):
            label, description = FEATURE_INFO[feature]
            with columns[index % 3]:
                form_values[feature] = st.text_input(
                    f"{label} (`{feature}`)",
                    value="",
                    placeholder="Masukkan angka",
                    help=description,
                    key=f"input_{feature}"
                )
                st.caption(description)

        submitted = st.form_submit_button(
            "Prediksi cluster",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not country_name.strip():
            st.error("Nama negara wajib diisi.")
        else:
            try:
                row_values = {
                    feature: parse_form_value(form_values[feature], feature)
                    for feature in FEATURES
                }
                input_data = pd.DataFrame([row_values], columns=FEATURES)
                predicted_cluster = int(predict_clusters(input_data)[0])

                st.success("Prediksi berhasil dilakukan.")
                result_col1, result_col2 = st.columns(2)
                result_col1.metric("Negara", country_name.strip())
                result_col2.metric("Hasil clustering", f"Cluster {predicted_cluster}")

                st.info(
                    f"Negara **{country_name.strip()}** masuk ke **Cluster "
                    f"{predicted_cluster}** berdasarkan sembilan indikator yang "
                    "dimasukkan."
                )
                st.info(
    """
    **Interpretasi Hasil Clustering**

    Hasil prediksi mengelompokkan negara berdasarkan kemiripan
    sembilan indikator pembangunan yang mencakup aspek sosial,
    ekonomi, dan kesehatan.

    Pengelompokan ini membantu memberikan gambaran mengenai
    pola dan karakteristik pembangunan negara berdasarkan
    hasil analisis algoritma K-Means.
    """
)
                )

                output = input_data.copy()
                output.insert(0, "country", country_name.strip())
                output["cluster"] = predicted_cluster
                st.dataframe(output, use_container_width=True)
                st.download_button(
                    "Unduh hasil prediksi",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_satu_negara.csv",
                    mime="text/csv"
                )
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Prediksi gagal diproses: {error}")


with tab_csv:
    st.subheader("Prediksi cluster banyak negara")
    st.write(
        "Unggah file CSV yang memiliki sembilan kolom fitur sesuai nama "
        "kolom dataset. Kolom `country` bersifat opsional."
    )
    st.code(", ".join(FEATURES), language="text")

    with st.expander("Lihat contoh format CSV"):
        example = pd.DataFrame(
            [
                {
                    "country": "Contoh Negara",
                    "child_mort": 10.0,
                    "exports": 30.0,
                    "health": 5.0,
                    "imports": 35.0,
                    "income": 12000.0,
                    "inflation": 2.5,
                    "life_expec": 72.0,
                    "total_fer": 2.1,
                    "gdpp": 8000.0,
                }
            ]
        )
        st.dataframe(example, use_container_width=True)
        st.caption("Angka di atas hanya contoh format, bukan data negara aktual.")

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        help="Gunakan nama kolom fitur yang sama persis dengan dataset."
    )

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            st.write("Pratinjau data yang diunggah:")
            st.dataframe(uploaded_data.head(), use_container_width=True)

            clean_features = validate_numeric_dataframe(uploaded_data)
            predictions = predict_clusters(clean_features)

            results = uploaded_data.copy()
            results["cluster"] = predictions

            st.success(
                f"Prediksi berhasil untuk {len(results)} baris data."
            )
            st.subheader("Hasil clustering")
            st.dataframe(results, use_container_width=True)

            cluster_counts = (
                results["cluster"]
                .value_counts()
                .sort_index()
                .rename_axis("cluster")
                .reset_index(name="jumlah_negara")
            )
            st.subheader("Jumlah negara per cluster")
            st.bar_chart(
                cluster_counts.set_index("cluster"),
                y="jumlah_negara"
            )

            st.download_button(
                "Unduh hasil clustering CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="hasil_clustering_negara.csv",
                mime="text/csv"
            )
        except Exception as error:
            st.error(f"File CSV tidak dapat diproses: {error}")


with tab_info:
    st.subheader("Informasi model")
    st.write("**Algoritma:** K-Means")
    st.write(f"**Jumlah cluster:** {cluster_count}")
    st.write("**Jumlah fitur:** 9")
    st.write("**Fitur yang digunakan:**")
    st.code(", ".join(FEATURES), language="text")
    st.write(
        "**Preprocessing:** PowerTransformer kemudian StandardScaler. "
        "Aplikasi menggunakan `transform()` agar parameter preprocessing "
        "tetap sama dengan proses training."
    )

    if isinstance(metrics, dict) and metrics:
        st.subheader("Metrik evaluasi model")
        metric_aliases = {
            "Silhouette Score": ["silhouette", "silhouette_score"],
            "Davies-Bouldin Index": ["davies_bouldin", "davies_bouldin_score", "dbi"],
            "Calinski-Harabasz Score": [
                "calinski_harabasz",
                "calinski_harabasz_score",
                "ch_score",
            ],
        }
        metric_columns = st.columns(3)
        for col, (label, aliases) in zip(metric_columns, metric_aliases.items()):
            metric_value = next(
                (metrics[key] for key in aliases if key in metrics),
                None
            )
            if metric_value is not None:
                try:
                    col.metric(label, f"{float(metric_value):.4f}")
                except (TypeError, ValueError):
                    col.metric(label, str(metric_value))
            else:
                col.metric(label, "Tidak tersedia")
    else:
        st.info(
            "Metrik evaluasi tidak ditemukan di dalam model bundle. "
            "Metrik tetap dapat dilihat pada notebook tahap Evaluation."
        )

    st.subheader("Catatan interpretasi")
    st.write(
        "Cluster 0 dan Cluster 1 merupakan label yang dihasilkan algoritma "
        "K-Means. Label tersebut tidak memiliki urutan tingkat kesejahteraan. "
        "Untuk memberi nama atau menjelaskan karakteristik setiap cluster, "
        "bandingkan rata-rata sembilan fitur pada tabel profil cluster di notebook."
    )

st.divider()
st.caption(
    "Country Development Clustering | Proyek Akhir Data Science | "
    "Hasil clustering merupakan alat bantu analisis, bukan keputusan kebijakan otomatis."
)
