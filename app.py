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
    "child_mort", "exports", "health", "imports", "income",
    "inflation", "life_expec", "total_fer", "gdpp"
]

FEATURE_LABELS = {
    "child_mort": "Child Mortality",
    "exports": "Exports (% GDP)",
    "health": "Health (% GDP)",
    "imports": "Imports (% GDP)",
    "income": "Income per Capita",
    "inflation": "Inflation (%)",
    "life_expec": "Life Expectancy",
    "total_fer": "Total Fertility",
    "gdpp": "GDP per Capita"
}

# ============================================================
# 2. JUDUL APLIKASI
# ============================================================
st.title("🌍 Country Development Clustering")
st.write(
    "Aplikasi pengelompokan negara menggunakan algoritma K-Means "
    "berdasarkan sembilan indikator sosial, ekonomi, dan kesehatan."
)
st.caption(
    "Masukkan sembilan indikator untuk memperoleh cluster dan "
    "interpretasi tingkat pembangunan berdasarkan profil cluster penelitian."
)
st.divider()

# ============================================================
# 3. MEMUAT MODEL BUNDLE TERBARU
# ============================================================
@st.cache_resource
def load_bundle(model_path, modified_time):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File {path.name} tidak ditemukan. "
            "Letakkan file model satu folder dengan app.py."
        )

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise TypeError("Isi model bundle harus berupa dictionary.")

    required_keys = {
        "transformer", "scaler", "model", "features", "metrics",
        "cluster_profile", "cluster_interpretation", "cluster_labels"
    }
    missing = required_keys.difference(bundle.keys())
    if missing:
        raise ValueError(
            "Model bundle belum lengkap. Komponen yang belum ada: "
            + ", ".join(sorted(missing))
            + ". Unggah ulang file joblib terbaru dari Google Colab."
        )

    if list(bundle["features"]) != FEATURES:
        raise ValueError(
            "Urutan fitur dalam model bundle tidak sesuai dengan aplikasi. "
            f"Fitur bundle: {list(bundle['features'])}"
        )

    profile = bundle["cluster_profile"]
    if not isinstance(profile, pd.DataFrame):
        raise TypeError("'cluster_profile' harus berupa pandas DataFrame.")

    missing_profile_cols = [f for f in FEATURES if f not in profile.columns]
    if missing_profile_cols:
        raise ValueError(
            "Kolom pada cluster_profile tidak lengkap: "
            + ", ".join(missing_profile_cols)
        )

    labels = {int(k): str(v) for k, v in bundle["cluster_labels"].items()}
    interpretations = {
        int(k): v for k, v in bundle["cluster_interpretation"].items()
    }

    model_clusters = set(range(int(bundle["model"].n_clusters)))
    if set(labels) != model_clusters:
        raise ValueError(
            "Mapping cluster_labels tidak cocok dengan jumlah cluster model."
        )
    if set(interpretations) != model_clusters:
        raise ValueError(
            "Mapping cluster_interpretation tidak cocok dengan jumlah cluster model."
        )

    profile_index = set(int(i) for i in profile.index)
    if not model_clusters.issubset(profile_index):
        raise ValueError(
            "Index cluster_profile tidak mencakup seluruh cluster model."
        )

    bundle["cluster_labels"] = labels
    bundle["cluster_interpretation"] = interpretations
    return bundle


try:
    bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Gagal memuat model: {exc}")
    st.stop()

transformer = bundle["transformer"]
scaler = bundle["scaler"]
model = bundle["model"]
metrics = bundle["metrics"]
cluster_profile = bundle["cluster_profile"]
cluster_labels = bundle["cluster_labels"]
cluster_interpretation = bundle["cluster_interpretation"]
reference_data = bundle.get("reference_data")  # tabel negara asli (opsional)

st.success(
    f"Model berhasil dimuat | K-Means | "
    f"Jumlah cluster: {model.n_clusters} | Profil cluster tersedia"
)

# ============================================================
# 4. FUNGSI VALIDASI DAN PREDIKSI
# ============================================================
def validate_input(dataframe):
    missing = [col for col in FEATURES if col not in dataframe.columns]
    if missing:
        raise ValueError("Kolom belum tersedia: " + ", ".join(missing))

    clean = dataframe[FEATURES].copy()
    for col in FEATURES:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    if clean.isna().any().any():
        bad = clean.columns[clean.isna().any()].tolist()
        raise ValueError(
            "Terdapat nilai kosong atau bukan angka pada: " + ", ".join(bad)
        )

    values = clean.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Data mengandung angka tak hingga atau tidak valid.")

    nonnegative = [f for f in FEATURES if f != "inflation"]
    negative = [f for f in nonnegative if (clean[f] < 0).any()]
    if negative:
        raise ValueError(
            "Nilai tidak boleh negatif pada: " + ", ".join(negative)
            + ". Inflation boleh bernilai negatif."
        )

    return clean


def predict_clusters(dataframe):
    clean = validate_input(dataframe)
    transformed = transformer.transform(clean)
    scaled = scaler.transform(transformed)
    return model.predict(scaled).astype(int)


@st.cache_data
def get_reference_scaled(_reference, _version):
    """Skala-kan data referensi sekali saja dengan transformer & scaler training."""
    ref_features = _reference[FEATURES].astype(float)
    return scaler.transform(transformer.transform(ref_features))


def find_similar_countries(dataframe, top_k=3):
    """Cari negara pada dataset penelitian yang paling mirip dengan input
    (jarak Euclidean pada ruang fitur yang sudah ditransformasi & diskalakan)."""
    if reference_data is None:
        return None

    clean = validate_input(dataframe)
    x_scaled = scaler.transform(transformer.transform(clean))
    ref_scaled = get_reference_scaled(reference_data, len(reference_data))

    distances = np.linalg.norm(ref_scaled - x_scaled[0], axis=1)
    idx = np.argsort(distances)[:top_k]

    result = reference_data.iloc[idx].copy()
    result.insert(1, "jarak", distances[idx])
    return result.reset_index(drop=True)


def show_similar_countries(input_df):
    st.subheader("Negara yang Sesuai dengan Indikator")
    similar = find_similar_countries(input_df, top_k=3)

    if similar is None:
        st.warning(
            "Data referensi negara belum ada di model bundle. Jalankan sel "
            "tambahan di Colab (reference_data) lalu unggah ulang file joblib."
        )
        return

    best = similar.iloc[0]
    if best["jarak"] < 0.05:
        st.success(f"Indikator ini cocok dengan negara: **{best['country']}**")
    else:
        st.info(
            f"Negara paling mirip dalam dataset: **{best['country']}** "
            f"(jarak {best['jarak']:.3f})"
        )

    show = similar.rename(columns={
        "country": "Negara", "jarak": "Jarak (makin kecil makin mirip)",
        "cluster": "Cluster", **FEATURE_LABELS
    })
    st.dataframe(show.round(3), hide_index=True, use_container_width=True)
    st.caption(
        "Model K-Means hanya menghasilkan cluster. Nama negara dicari dengan "
        "membandingkan input terhadap data negara pada dataset penelitian."
    )


def show_cluster_interpretation(cluster_id):
    cluster_id = int(cluster_id)
    category = cluster_labels[cluster_id]
    interpretation = cluster_interpretation[cluster_id]

    if isinstance(interpretation, dict):
        description = interpretation.get("deskripsi", "")
        category = interpretation.get("kategori", category)
    else:
        description = str(interpretation)

    if "Tinggi" in category:
        st.success(f"**Kategori pembangunan: {category}**")
    elif "Rendah" in category:
        st.warning(f"**Kategori pembangunan: {category}**")
    else:
        st.info(f"**Kategori cluster: {category}**")

    if description:
        st.write(description)

    st.markdown("**Dasar interpretasi: profil rata-rata cluster penelitian**")
    profile_row = cluster_profile.loc[cluster_id, FEATURES]
    profile_table = pd.DataFrame({
        "Indikator": [FEATURE_LABELS[f] for f in FEATURES],
        "Rata-rata cluster": [
            float(profile_row[f]) for f in FEATURES
        ]
    })
    st.dataframe(profile_table, hide_index=True, use_container_width=True)

    st.caption(
        "Kategori tinggi/rendah mengikuti mapping interpretasi hasil penelitian "
        "yang disimpan dalam model bundle, bukan ditentukan dari nomor cluster."
    )


# ============================================================
# 5. TAB APLIKASI
# ============================================================
tab_single, tab_csv, tab_info = st.tabs([
    "Prediksi Satu Negara",
    "Prediksi Banyak Negara (CSV)",
    "Informasi Model"
])

# ============================================================
# 6. PREDIKSI SATU NEGARA
# ============================================================
with tab_single:
    st.subheader("Prediksi Tingkat Pembangunan")
    st.write(
        "Masukkan sembilan indikator negara. Aplikasi akan memprediksi "
        "cluster dan menampilkan interpretasi tingkat pembangunan "
        "berdasarkan profil cluster dari penelitian."
    )

    with st.form("single_country_form"):
        st.markdown("**Sembilan indikator input**")
        st.caption(
            "Gunakan satuan dan skala yang sama dengan dataset training. "
            "Gunakan titik atau koma untuk desimal; inflation boleh negatif."
        )

        country_name = st.text_input(
            "Nama Negara (opsional)",
            placeholder="Contoh: Japan",
            key="input_country"
        )

        columns = st.columns(3)
        raw_values = {}

        for i, feature in enumerate(FEATURES):
            with columns[i % 3]:
                raw_values[feature] = st.text_input(
                    FEATURE_LABELS[feature],
                    placeholder="Masukkan angka",
                    key=f"input_{feature}"
                )

        submitted = st.form_submit_button(
            "Prediksi Pembangunan Negara",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            values = {}
            for feature in FEATURES:
                raw = raw_values[feature].strip()
                if not raw:
                    raise ValueError(
                        f"Indikator {FEATURE_LABELS[feature]} belum diisi."
                    )
                try:
                    values[feature] = float(raw.replace(",", "."))
                except ValueError:
                    raise ValueError(
                        f"{FEATURE_LABELS[feature]} harus berupa angka."
                    )

            input_df = pd.DataFrame([values], columns=FEATURES)
            predicted_cluster = int(predict_clusters(input_df)[0])

            st.divider()
            st.subheader("Hasil Prediksi")

            country_name = country_name.strip()
            col0, col1 = st.columns(2)
            col0.metric("Negara", country_name if country_name else "-")
            col1.metric("Cluster Negara", f"Cluster {predicted_cluster}")
            st.metric(
                "Kategori Pembangunan",
                cluster_labels[predicted_cluster]
            )

            # Jika nama negara tidak diisi, cari negara paling mirip
            if not country_name:
                show_similar_countries(input_df)

            st.subheader("Kesimpulan dan Interpretasi")
            show_cluster_interpretation(predicted_cluster)

            st.markdown("**Data indikator yang dimasukkan**")
            display_input = input_df.rename(columns=FEATURE_LABELS)
            st.dataframe(display_input, hide_index=True, use_container_width=True)

            result_df = input_df.copy()
            result_df.insert(0, "cluster", predicted_cluster)
            result_df.insert(
                1, "kategori_pembangunan",
                cluster_labels[predicted_cluster]
            )
            if country_name:
                result_df.insert(0, "country", country_name)
            else:
                similar_df = find_similar_countries(input_df, top_k=1)
                if similar_df is not None:
                    result_df.insert(
                        0, "negara_paling_mirip", similar_df.loc[0, "country"]
                    )
            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh hasil prediksi",
                data=csv_data,
                file_name="hasil_prediksi_country_clustering.csv",
                mime="text/csv"
            )

        except ValueError as exc:
            st.error(f"Input tidak valid: {exc}")
        except Exception as exc:
            st.error(f"Prediksi gagal diproses: {exc}")

# ============================================================
# 7. PREDIKSI BANYAK NEGARA DENGAN CSV
# ============================================================
with tab_csv:
    st.subheader("Prediksi Banyak Negara")
    st.write(
        "Unggah CSV dengan sembilan kolom indikator. Kolom country bersifat "
        "opsional dan akan dipertahankan jika tersedia."
    )
    st.code(", ".join(["country"] + FEATURES))

    uploaded_file = st.file_uploader(
        "Unggah file CSV",
        type=["csv"],
        key="multi_country_csv"
    )

    if uploaded_file is not None:
        try:
            source_df = pd.read_csv(uploaded_file)
            clean_df = validate_input(source_df)
            predictions = predict_clusters(clean_df)

            result_df = source_df.copy()
            result_df["cluster"] = predictions
            result_df["kategori_pembangunan"] = [
                cluster_labels[int(cluster)] for cluster in predictions
            ]

            st.success(
                f"Prediksi berhasil untuk {len(result_df)} baris data."
            )
            st.dataframe(result_df, use_container_width=True)

            st.subheader("Distribusi Kategori Pembangunan")
            distribution = (
                result_df["kategori_pembangunan"]
                .value_counts()
                .rename_axis("Kategori")
                .to_frame("Jumlah Negara")
            )
            st.dataframe(distribution, use_container_width=True)

            st.subheader("Profil Cluster pada Hasil Penelitian")
            st.dataframe(cluster_profile.round(2), use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh hasil clustering CSV",
                data=csv_data,
                file_name="hasil_clustering_negara.csv",
                mime="text/csv"
            )

        except Exception as exc:
            st.error(f"File CSV gagal diproses: {exc}")

# ============================================================
# 8. INFORMASI DAN EVALUASI MODEL
# ============================================================
with tab_info:
    st.subheader("Informasi dan Evaluasi Model")

    metric_definitions = [
        ("Silhouette Score", "silhouette"),
        ("Davies-Bouldin Index", "davies_bouldin"),
        ("Calinski-Harabasz Score", "calinski_harabasz")
    ]
    metric_cols = st.columns(3)

    for col, (label, key) in zip(metric_cols, metric_definitions):
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if value is not None:
            col.metric(label, f"{float(value):.4f}")
        else:
            col.metric(label, "Tidak tersedia")

    st.divider()
    st.markdown("**Algoritma**")
    st.write("K-Means")

    st.markdown("**Fitur yang digunakan**")
    st.write(FEATURES)

    st.markdown("**Preprocessing**")
    st.write(
        "PowerTransformer metode Yeo-Johnson dilanjutkan StandardScaler, "
        "menggunakan objek preprocessing yang disimpan dari proses training."
    )

    st.markdown("**Profil rata-rata cluster penelitian**")
    st.dataframe(cluster_profile.round(2), use_container_width=True)

    st.markdown("**Interpretasi cluster**")
    for cluster_id in sorted(cluster_labels):
        st.markdown(f"**Cluster {cluster_id}: {cluster_labels[cluster_id]}**")
        info = cluster_interpretation[cluster_id]
        if isinstance(info, dict):
            st.write(info.get("deskripsi", ""))
        else:
            st.write(str(info))

    st.info(
        "Kategori pembangunan merupakan interpretasi relatif berdasarkan "
        "profil rata-rata cluster pada dataset penelitian. Kategori ini "
        "bukan peringkat resmi pembangunan negara."
    )

# ============================================================
# 9. FOOTER
# ============================================================
st.divider()
st.caption(
    "Country Development Clustering | Final Project Data Science | "
    "CRISP-DM - Deployment"
)
