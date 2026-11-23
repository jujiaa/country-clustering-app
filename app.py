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
                )

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
        except ValueError as exc:
            st.error(f"Input belum valid: {exc}")
        except Exception as exc:
            st.error(f"Prediksi gagal diproses: {exc}")

with tab2:
    st.subheader("Unggah file CSV")
    st.write(
        "CSV harus memiliki sembilan kolom fitur dengan nama persis seperti "
        "yang tercantum di bawah. Kolom `country` boleh disertakan dan akan "
        "dipertahankan sebagai identitas."
    )
    st.code(", ".join(FEATURES))
    st.caption("Nilai inflation boleh negatif; delapan fitur lainnya harus nol atau positif.")
    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
            missing = [f for f in FEATURES if f not in input_df.columns]
            if missing:
                st.error(f"Kolom fitur berikut tidak ditemukan: {missing}")
            else:
                model_input = input_df[FEATURES].copy()
                for col in FEATURES:
                    model_input[col] = pd.to_numeric(model_input[col], errors="coerce")

                if model_input.isna().any().any():
                    bad_cols = model_input.columns[model_input.isna().any()].tolist()
                    st.error(
                        "Terdapat nilai kosong atau non-numerik pada kolom: "
                        f"{bad_cols}. Perbaiki CSV terlebih dahulu."
                    )
                elif not np.isfinite(model_input.to_numpy(dtype=float)).all():
                    st.error("CSV mengandung nilai tak hingga atau bukan angka yang valid.")
                else:
                    nonnegative_features = [f for f in FEATURES if f != "inflation"]
                    negative_cols = [
                        f for f in nonnegative_features
                        if (model_input[f] < 0).any()
                    ]
                    if negative_cols:
                        st.error(
                            "Nilai negatif tidak diperbolehkan pada kolom: "
                            f"{negative_cols}. Kolom inflation boleh negatif."
                        )
                    else:
                        transformed = transformer.transform(model_input)
                        scaled = scaler.transform(transformed)
                        result = input_df.copy()
                        result["cluster"] = model.predict(scaled).astype(int)
                        st.success(f"Prediksi selesai untuk {len(result)} baris.")
                        st.dataframe(result, use_container_width=True)
                        csv_bytes = result.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Unduh hasil prediksi CSV",
                            data=csv_bytes,
                            file_name="hasil_clustering_negara.csv",
                            mime="text/csv"
                        )
        except Exception as exc:
            st.error(f"CSV gagal diproses: {exc}")

with tab3:
    st.subheader("Informasi dan evaluasi model")
    metrics = bundle["metrics"]
    a, b, c = st.columns(3)
    a.metric("Silhouette Score", f"{metrics['silhouette']:.4f}")
    b.metric("Davies-Bouldin Index", f"{metrics['davies_bouldin']:.4f}")
    c.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.4f}")
    st.write("**Fitur yang digunakan**")
    st.write(FEATURES)
    st.write(
        "**Preprocessing:** PowerTransformer metode Yeo-Johnson, "
        "dilanjutkan StandardScaler."
    )
    st.write("**Model:** K-Means dengan random_state=42 dan n_init=10.")
    st.warning(
        "Pastikan file country_clustering_model.joblib berasal dari proses training "
        "notebook proyek ini. Jangan menggunakan bundle dari dataset atau "
        "urutan fitur yang berbeda."
    )
