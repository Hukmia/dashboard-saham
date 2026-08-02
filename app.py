"""
DASHBOARD ANALISIS SAHAM
========================
Fitur:
1. Upload CSV data harga saham (untuk prediksi GRU & XGBoost)
2. Upload data fundamental (satu file Excel/CSV per perusahaan, kolom:
   Waktu, EPS, ROA, ROE, CR, DER, PER) untuk analisis & perbandingan rasio
3. Prediksi harga saham menggunakan model GRU (Deep Learning) dan
   XGBoost (Machine Learning, berbasis log-return), lengkap dengan
   perbandingan performa kedua model

Cara pakai (dari Google Colab): lihat file colab_runner.txt
Cara deploy permanen: upload ke GitHub, lalu deploy via Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Dashboard Analitik Saham", page_icon="📊", layout="wide")

# =========================================================
# TEMA PROFESIONAL (DARK MODE + CUSTOM CSS)
# =========================================================
st.markdown(
    """
    <style>
        .dashboard-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.1rem;
        }
        .dashboard-header .icon-box {
            font-size: 1.8rem;
            background: linear-gradient(135deg, #FF6B4A, #FF8C42);
            padding: 0.35rem 0.55rem;
            border-radius: 10px;
        }
        .dashboard-header h1 {
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
        }
        .dashboard-subtitle {
            color: #9CA3AF;
            font-size: 0.95rem;
            margin-bottom: 1.3rem;
        }

        /* Kartu metrik */
        div[data-testid="stMetric"] {
            background-color: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.9rem 1.1rem 0.7rem 1.1rem;
        }
        div[data-testid="stMetricLabel"] {
            color: #9CA3AF !important;
            font-size: 0.82rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            font-weight: 700 !important;
        }

        /* Tab navigasi utama */
        button[data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 600;
            padding: 0.55rem 1.1rem;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #FF6B4A !important;
            border-bottom: 3px solid #FF6B4A !important;
        }

        /* Tombol */
        .stButton>button, .stDownloadButton>button {
            border-radius: 8px;
            font-weight: 600;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        hr { margin: 0.8rem 0 1.2rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dashboard-header">
        <span class="icon-box">📊</span>
        <h1>Dashboard Analitik Fundamental &amp; Prediksi Saham</h1>
    </div>
    <div class="dashboard-subtitle">
        Analisis rasio keuangan, prediksi harga saham (GRU &amp; XGBoost), dan rekomendasi KPI —
        sektor Consumer Non-Cyclicals di Indonesia.
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR — BRANDING & INFO
# =========================================================
with st.sidebar:
    st.markdown("### 📊 Dashboard Saham")
    st.caption("Analitik Fundamental & Prediksi (GRU + XGBoost)")
    st.divider()
    st.markdown(
        "**Alur pemakaian:**\n"
        "1. Upload data di tab *Upload Data*\n"
        "2. Lihat rasio di tab *Analisis Rasio Keuangan*\n"
        "3. Latih model di tab *Prediksi Harga*\n"
        "4. Lihat rekomendasi di tab *Ringkasan & KPI*"
    )
    if st.session_state.get("df_harga") is not None:
        st.divider()
        st.caption(f"📈 Data harga aktif: **{len(st.session_state.df_harga)} baris**")
    if st.session_state.get("df_keuangan"):
        st.caption(f"🏢 Perusahaan fundamental: **{len(st.session_state.df_keuangan)}**")

# =========================================================
# NAVIGASI TAB
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📁  Upload Data",
        "📊  Analisis Rasio Keuangan",
        "🤖  Prediksi Harga (GRU & XGBoost)",
        "✅  Ringkasan & Rekomendasi KPI",
    ]
)

if "df_harga" not in st.session_state:
    st.session_state.df_harga = None
if "df_keuangan" not in st.session_state:
    st.session_state.df_keuangan = None
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "forecast_results" not in st.session_state:
    st.session_state.forecast_results = {}   # {"GRU": {...}, "XGBoost": {...}}
if "eval_results" not in st.session_state:
    st.session_state.eval_results = {}       # {"GRU": {"rmse":.., "mae":.., "mape":..}, "XGBoost": {...}}


# =========================================================
# HALAMAN 1 — UPLOAD DATA
# =========================================================
with tab1:
    st.title("Upload Data")

    st.subheader("A. Data Harga Saham (harian)")
    st.caption("Kolom wajib: **Date, Close**. Kolom lain (Open, High, Low, Volume) opsional.")
    file_harga = st.file_uploader("Upload CSV harga saham", type=["csv"], key="upload_harga")

    if file_harga is not None:
        df = pd.read_csv(file_harga)
        df.columns = [c.strip() for c in df.columns]
        if "Date" not in df.columns or "Close" not in df.columns:
            st.error("CSV harus punya kolom 'Date' dan 'Close'.")
        else:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            df = df.dropna(subset=["Close"]).reset_index(drop=True)
            st.session_state.df_harga = df
            st.success(f"Berhasil upload {len(df)} baris data harga.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Jumlah Data", f"{len(df):,}")
            m2.metric("Periode Awal", df["Date"].min().strftime("%d %b %Y"))
            m3.metric("Periode Akhir", df["Date"].max().strftime("%d %b %Y"))
            m4.metric("Harga Terakhir", f"Rp {df['Close'].iloc[-1]:,.0f}")

            st.dataframe(df.tail(10), use_container_width=True)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df["Date"], df["Close"])
            ax.set_title("Preview Harga Saham")
            ax.grid(True)
            st.pyplot(fig)

    st.divider()

    st.subheader("B. Data Fundamental / Rasio Keuangan (satu file per perusahaan)")
    st.caption(
        "Upload **satu file per perusahaan** (Excel .xlsx atau CSV). "
        "Tiap file punya kolom **Waktu** (mis. tahun: 2021, 2022, ...) dan kolom rasio: "
        "**EPS, ROA, ROE, CR, DER, PER**."
    )
    files_fundamental = st.file_uploader(
        "Upload file data fundamental (bisa pilih banyak file sekaligus)",
        type=["xlsx", "csv"],
        accept_multiple_files=True,
        key="upload_fundamental",
    )

    RATIO_NAMES = ["EPS", "ROA", "ROE", "CR", "DER", "PER"]
    PERCENT_RATIOS = ["ROA", "ROE"]  # rasio yang biasanya dalam bentuk persen

    def normalisasi_nama_kolom(nama):
        """Samakan nama kolom biar cocok walau ada spasi/simbol/huruf besar-kecil beda."""
        return (
            str(nama).strip().upper()
            .replace("(%)", "").replace("%", "")
            .replace(" ", "").replace("-", "").replace("_", "")
        )

    def bersihkan_kolom_angka(series, is_percent=False):
        """Bersihkan kolom rasio: hapus '%', tangani None/kosong, ubah ke angka.
        Untuk kolom persen: kalau nilainya pecahan (mis. 0.1542) ubah ke skala persen (15.42)."""
        series = series.astype(str).str.replace("%", "", regex=False).str.strip()
        series = series.replace({"None": None, "none": None, "nan": None, "NaN": None, "": None})
        series = pd.to_numeric(series, errors="coerce")
        if is_percent:
            non_null = series.dropna()
            if len(non_null) > 0 and non_null.abs().max() <= 1:
                series = series * 100
        return series

    if files_fundamental:
        company_data = {}
        for f in files_fundamental:
            default_name = (
                f.name.rsplit(".", 1)[0]
                .upper()
                .replace("DATA SEMHAS", "")
                .replace("DATA", "")
                .strip()
            ) or f.name

            nama_perusahaan = st.text_input(
                f"Nama perusahaan untuk file '{f.name}'",
                value=default_name,
                key=f"nama_{f.name}",
            )

            try:
                if f.name.lower().endswith(".csv"):
                    fdf = pd.read_csv(f)
                else:
                    fdf = pd.read_excel(f, sheet_name=0)
            except Exception as e:
                st.error(f"Gagal membaca file '{f.name}': {e}")
                continue

            fdf.columns = [str(c).strip() for c in fdf.columns]

            # cocokkan nama kolom rasio secara fleksibel (tahan spasi/simbol/huruf besar-kecil)
            peta_kolom = {normalisasi_nama_kolom(c): c for c in fdf.columns}
            kolom_waktu_asli = peta_kolom.get("WAKTU")

            if kolom_waktu_asli is None:
                st.error(f"File '{f.name}' dilewati: tidak ada kolom 'Waktu'. Kolom terdeteksi: {list(fdf.columns)}")
                continue

            rename_map = {kolom_waktu_asli: "Waktu"}
            kolom_rasio_ada = []
            for rasio in RATIO_NAMES:
                if rasio in peta_kolom:
                    rename_map[peta_kolom[rasio]] = rasio
                    kolom_rasio_ada.append(rasio)

            if not kolom_rasio_ada:
                st.error(f"File '{f.name}' dilewati: tidak ada kolom rasio yang dikenal. Kolom terdeteksi: {list(fdf.columns)}")
                continue

            fdf = fdf.rename(columns=rename_map)
            fdf = fdf[["Waktu"] + kolom_rasio_ada].dropna(subset=["Waktu"]).reset_index(drop=True)

            for kol in kolom_rasio_ada:
                fdf[kol] = bersihkan_kolom_angka(fdf[kol], is_percent=(kol in PERCENT_RATIOS))

            kolom_kosong = [kol for kol in kolom_rasio_ada if fdf[kol].isna().all()]
            if kolom_kosong:
                st.warning(f"File '{f.name}': kolom {kolom_kosong} kosong semua setelah dibersihkan — cek isi filenya.")

            company_data[nama_perusahaan] = fdf

        if company_data:
            st.session_state.df_keuangan = company_data
            st.success(
                f"Berhasil upload data fundamental {len(company_data)} perusahaan: "
                + ", ".join(company_data.keys())
            )
            for nama, fdf in company_data.items():
                with st.expander(f"Preview data {nama}"):
                    st.dataframe(fdf, use_container_width=True)

    with st.expander("Contoh format file data fundamental (1 file = 1 perusahaan)"):
        contoh = pd.DataFrame({
            "Waktu": [2021, 2022, 2023, 2024, 2025],
            "EPS": [12.13, 2.95, 0.09, 12.80, 27.22],
            "ROA": ["8.25%", "2.31%", "0.06%", "8.17%", "15.42%"],
            "ROE": ["43.87%", "11.15%", "0.20%", "22.32%", "33.38%"],
            "CR": [1.10, 0.98, 0.94, 1.15, 1.39],
            "DER": [3.85, 3.34, 1.98, 1.46, 0.91],
            "PER": [0.00, 0.00, 1966.67, 21.56, 33.98],
        })
        st.dataframe(contoh, use_container_width=True)
        st.caption("Nama file bebas — nama perusahaan bisa kamu ubah manual di kotak isian setelah upload.")


# =========================================================
# HALAMAN 2 — ANALISIS RASIO KEUANGAN
# =========================================================
with tab2:
    st.title("Analisis Rasio Keuangan")

    company_data = st.session_state.df_keuangan
    if not company_data:
        st.warning("Silakan upload data fundamental dulu di halaman '1. Upload Data'.")
    else:
        RATIO_ORDER = ["EPS", "ROA", "ROE", "CR", "DER", "PER"]
        RATIO_LABEL = {
            "EPS": "Earning per Share (EPS)",
            "ROA": "Return on Assets - ROA (%)",
            "ROE": "Return on Equity - ROE (%)",
            "CR": "Current Ratio (CR)",
            "DER": "Debt to Equity Ratio (DER)",
            "PER": "Price to Earnings Ratio (PER)",
        }

        semua_perusahaan = sorted(company_data.keys())
        perusahaan_dipilih = st.multiselect(
            "Pilih perusahaan yang mau dibandingkan",
            options=semua_perusahaan,
            default=semua_perusahaan,
        )

        if not perusahaan_dipilih:
            st.info("Pilih minimal satu perusahaan untuk menampilkan grafik.")
        else:
            # gabungkan data semua perusahaan terpilih jadi 1 dict per rasio: {rasio: df(Waktu, perusahaan1, perusahaan2, ...)}
            rasio_tersedia = [r for r in RATIO_ORDER if any(r in company_data[p].columns for p in perusahaan_dipilih)]

            tabs = st.tabs(rasio_tersedia)

            for tab, nama_rasio in zip(tabs, rasio_tersedia):
                with tab:
                    gabungan = None
                    for perusahaan in perusahaan_dipilih:
                        pdf = company_data[perusahaan]
                        if nama_rasio not in pdf.columns:
                            continue
                        kolom = pdf[["Waktu", nama_rasio]].rename(columns={nama_rasio: perusahaan})
                        gabungan = kolom if gabungan is None else pd.merge(gabungan, kolom, on="Waktu", how="outer")

                    if gabungan is None or gabungan.empty:
                        st.info(f"Tidak ada data {nama_rasio} untuk perusahaan yang dipilih.")
                        continue

                    gabungan = gabungan.sort_values("Waktu").reset_index(drop=True)
                    kolom_perusahaan = [c for c in gabungan.columns if c != "Waktu"]

                    st.subheader(RATIO_LABEL.get(nama_rasio, nama_rasio))
                    fig, ax = plt.subplots(figsize=(11, 4.5))
                    for perusahaan in kolom_perusahaan:
                        ax.plot(
                            gabungan["Waktu"].astype(str), gabungan[perusahaan],
                            marker="o", markersize=4, label=perusahaan
                        )
                    ax.set_title(f"Perbandingan {nama_rasio}")
                    ax.set_xlabel("Waktu")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                    with st.expander(f"Lihat tabel data {nama_rasio}"):
                        st.dataframe(gabungan, use_container_width=True)

                    csv_download = gabungan.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"⬇️ Download data {nama_rasio} (CSV)",
                        csv_download,
                        f"{nama_rasio.lower()}_perbandingan.csv",
                        "text/csv",
                        key=f"download_{nama_rasio}",
                    )

            st.divider()
            st.subheader("Ringkasan Nilai Terbaru (Periode Terakhir per Rasio)")
            ringkasan = []
            for perusahaan in perusahaan_dipilih:
                pdf = company_data[perusahaan]
                baris_terakhir = pdf.sort_values("Waktu").iloc[-1]
                baris = {"Perusahaan": perusahaan, "Waktu": baris_terakhir["Waktu"]}
                for r in RATIO_ORDER:
                    if r in pdf.columns:
                        baris[r] = baris_terakhir[r]
                ringkasan.append(baris)

            if ringkasan:
                ringkasan_df = pd.DataFrame(ringkasan).set_index("Perusahaan")
                st.dataframe(ringkasan_df.round(2), use_container_width=True)


# =========================================================
# HALAMAN 3 — PREDIKSI GRU & XGBOOST
# =========================================================
with tab3:
    st.title("Prediksi Harga Saham")

    df = st.session_state.df_harga
    if df is None:
        st.warning("Silakan upload data harga saham dulu di halaman '1. Upload Data'.")
    else:
        algo = st.radio(
            "Pilih Algoritma",
            ["GRU (Deep Learning)", "XGBoost (Machine Learning)"],
            horizontal=True,
        )

        # =====================================================
        # ALGORITMA: GRU
        # =====================================================
        if algo == "GRU (Deep Learning)":
            st.subheader("Pengaturan Model GRU")

            future_days = st.slider(
                "🗓️ Rentang Waktu Prediksi ke Depan (hari kerja)",
                min_value=5, max_value=180, value=30, step=5, key="gru_future"
            )
            st.caption(f"Model akan memprediksi **{future_days} hari kerja** ke depan dari data terakhir.")

            c1, c2, c3 = st.columns(3)
            with c1:
                window_size = st.number_input("Window size (hari)", 5, 120, 30, key="gru_window")
            with c2:
                hidden_size = st.number_input("Hidden units GRU", 8, 256, 64, key="gru_hidden")
            with c3:
                epochs = st.number_input("Epochs", 5, 300, 50, key="gru_epochs")

            train_button = st.button("🚀 Latih Model GRU & Prediksi", type="primary", key="btn_gru")

            if train_button:
                if len(df) < window_size + 20:
                    st.error("Data terlalu sedikit untuk window size ini. Tambah data atau kecilkan window size.")
                else:
                    with st.spinner("Melatih model GRU, mohon tunggu..."):
                        import tensorflow as tf
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import GRU, Dense, Dropout
                        from tensorflow.keras.optimizers import Adam
                        from tensorflow.keras import backend as K

                        seed = 42
                        np.random.seed(seed)
                        tf.random.set_seed(seed)

                        total = len(df)
                        train_size = int(total * 0.60)
                        val_size = int(total * 0.20)

                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaler.fit(df[["Close"]])
                        df["Close_Normalized"] = scaler.transform(df[["Close"]])

                        values = df["Close_Normalized"].values
                        X_all, y_all, target_idx = [], [], []
                        for i in range(len(values) - window_size):
                            X_all.append(values[i:i + window_size])
                            y_all.append(values[i + window_size])
                            target_idx.append(i + window_size)

                        X_all = np.array(X_all)
                        y_all = np.array(y_all)
                        target_idx = np.array(target_idx)

                        train_mask = target_idx < train_size
                        val_mask = (target_idx >= train_size) & (target_idx < train_size + val_size)
                        test_mask = target_idx >= train_size + val_size

                        X_train, y_train = X_all[train_mask], y_all[train_mask]
                        X_val, y_val = X_all[val_mask], y_all[val_mask]
                        X_test, y_test = X_all[test_mask], y_all[test_mask]

                        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                        K.clear_session()
                        model = Sequential(name="Model_GRU")
                        model.add(GRU(int(hidden_size), return_sequences=True, input_shape=(window_size, 1)))
                        model.add(Dropout(0.2))
                        model.add(GRU(int(hidden_size)))
                        model.add(Dropout(0.2))
                        model.add(Dense(1))
                        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

                        history = model.fit(
                            X_train, y_train,
                            epochs=int(epochs),
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            verbose=0
                        )

                        st.session_state.model = model
                        st.session_state.scaler = scaler

                    st.success("Model GRU selesai dilatih!")

                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(history.history["loss"], label="Train Loss")
                    ax.plot(history.history["val_loss"], label="Validation Loss")
                    ax.set_title("Train vs Validation Loss - GRU")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    if len(X_test) > 0:
                        y_pred_test = model.predict(X_test, verbose=0).reshape(-1)
                        y_true_test = y_test.reshape(-1)
                        rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
                        mae = mean_absolute_error(y_true_test, y_pred_test)
                        mape = np.mean(np.abs((y_true_test - y_pred_test) / (y_true_test + 1e-10))) * 100

                        m1, m2, m3 = st.columns(3)
                        m1.metric("RMSE", f"{rmse:.4f}")
                        m2.metric("MAE", f"{mae:.4f}")
                        m3.metric("MAPE", f"{mape:.2f}%")

                        st.session_state.eval_results["GRU"] = {"rmse": rmse, "mae": mae, "mape": mape}
                    else:
                        st.info("Data test kosong (data terlalu sedikit) — evaluasi dilewati.")

                    # forecast rekursif ke depan
                    last_sequence = values[-window_size:].reshape(window_size, 1)
                    current_input = last_sequence.copy()
                    future_predictions_scaled = []
                    for _ in range(int(future_days)):
                        pred = model.predict(current_input.reshape(1, window_size, 1), verbose=0)
                        future_predictions_scaled.append(pred[0, 0])
                        current_input = np.roll(current_input, -1, axis=0)
                        current_input[-1] = pred

                    future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
                    future_predictions = scaler.inverse_transform(future_predictions_scaled).flatten()

                    last_date = df["Date"].iloc[-1]
                    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

                    st.session_state.forecast_results["GRU"] = {
                        "future_dates": future_dates,
                        "future_predictions": future_predictions,
                        "last_actual_price": float(df["Close"].iloc[-1]),
                        "last_actual_date": df["Date"].iloc[-1],
                    }
                    # kompatibilitas dengan halaman KPI versi lama
                    st.session_state.forecast_result = st.session_state.forecast_results["GRU"]

                    st.subheader("Hasil Prediksi - GRU")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df["Date"], df["Close"], label="Historical Prices")
                    ax.plot(future_dates, future_predictions, linestyle="--", color="#DD8452", label="Prediction (GRU)")
                    ax.scatter(future_dates[-1], future_predictions[-1], color="red", zorder=5)
                    ax.text(future_dates[-1], future_predictions[-1], f"{future_predictions[-1]:.2f}")
                    ax.set_title("Prediksi Harga Saham - GRU")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    hasil_forecast = pd.DataFrame({
                        "Date": future_dates,
                        "Predicted_Close": future_predictions
                    })
                    st.dataframe(hasil_forecast, use_container_width=True)

                    csv_forecast = hasil_forecast.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download hasil prediksi GRU (CSV)", csv_forecast, "prediksi_gru.csv", "text/csv", key="dl_gru")

                    st.caption(
                        "⚠️ Catatan: forecast rekursif jangka panjang (>30-60 hari) cenderung "
                        "makin tidak akurat karena error terakumulasi setiap langkah. "
                        "Gunakan hasil ini sebagai gambaran tren, bukan angka pasti."
                    )

        # =====================================================
        # ALGORITMA: XGBOOST
        # =====================================================
        else:
            st.subheader("Pengaturan Model XGBoost")
            st.caption(
                "Model diproses menggunakan pendekatan **log-return** (bukan harga absolut), karena "
                "XGBoost sebagai model berbasis pohon keputusan tidak dapat melakukan ekstrapolasi "
                "nilai di luar rentang data pelatihan. Log-return bersifat lebih stasioner sehingga "
                "prediksi menjadi lebih stabil (Tsay, 2010)."
            )

            future_days_xgb = st.slider(
                "🗓️ Rentang Waktu Prediksi ke Depan (hari kerja)",
                min_value=5, max_value=180, value=30, step=5, key="xgb_future"
            )
            st.caption(f"Model akan memprediksi **{future_days_xgb} hari kerja** ke depan dari data terakhir.")

            c1, c2 = st.columns(2)
            with c1:
                window_size_xgb = st.number_input("Window size (hari)", 5, 120, 30, key="xgb_window")
            with c2:
                bo_iter = st.number_input("Iterasi Bayesian Optimization", 5, 40, 15, key="xgb_bo_iter")

            train_button_xgb = st.button("🚀 Latih Model XGBoost & Prediksi", type="primary", key="btn_xgb")

            if train_button_xgb:
                if len(df) < window_size_xgb + future_days_xgb + 20:
                    st.error("Data terlalu sedikit untuk window size & horizon ini. Tambah data atau kecilkan angkanya.")
                else:
                    with st.spinner("Melatih model XGBoost (Bayesian Optimization), mohon tunggu..."):
                        from xgboost import XGBRegressor
                        from sklearn.multioutput import MultiOutputRegressor
                        from bayes_opt import BayesianOptimization

                        seed = 42
                        np.random.seed(seed)

                        total = len(df)
                        train_size = int(total * 0.60)
                        val_size = int(total * 0.20)

                        # --- log-return (bukan harga absolut) ---
                        log_return = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
                        scaler_return = MinMaxScaler(feature_range=(0, 1))
                        log_return_scaled = scaler_return.fit_transform(log_return.reshape(-1, 1)).flatten()

                        X_all, y_all, target_idx = [], [], []
                        for i in range(len(log_return_scaled) - window_size_xgb):
                            X_all.append(log_return_scaled[i:i + window_size_xgb])
                            y_all.append(log_return_scaled[i + window_size_xgb])
                            target_idx.append(i + window_size_xgb)

                        X_all = np.array(X_all)
                        y_all = np.array(y_all)
                        target_idx = np.array(target_idx)

                        train_mask = target_idx < train_size
                        val_mask = (target_idx >= train_size) & (target_idx < train_size + val_size)
                        test_mask = target_idx >= train_size + val_size

                        X_train_x, y_train_x = X_all[train_mask], y_all[train_mask]
                        X_val_x, y_val_x = X_all[val_mask], y_all[val_mask]
                        X_test_x, y_test_x = X_all[test_mask], y_all[test_mask]

                        # --- Bayesian Optimization hyperparameter ---
                        def train_xgb(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight):
                            model = XGBRegressor(
                                n_estimators=int(n_estimators), max_depth=int(max_depth),
                                learning_rate=learning_rate, subsample=subsample,
                                colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                                objective='reg:squarederror', random_state=seed, n_jobs=-1,
                            )
                            model.fit(X_train_x, y_train_x, eval_set=[(X_val_x, y_val_x)], verbose=False)
                            return -mean_squared_error(y_val_x, model.predict(X_val_x))

                        pbounds = {
                            'n_estimators': (50, 500), 'max_depth': (2, 10),
                            'learning_rate': (0.001, 0.3), 'subsample': (0.5, 1.0),
                            'colsample_bytree': (0.5, 1.0), 'min_child_weight': (1, 10),
                        }
                        bo = BayesianOptimization(f=train_xgb, pbounds=pbounds, random_state=seed, verbose=0)
                        bo.maximize(init_points=5, n_iter=int(bo_iter))
                        bp = bo.max['params']

                        n_estimators = int(bp['n_estimators'])
                        max_depth = int(bp['max_depth'])
                        learning_rate_xgb = float(bp['learning_rate'])
                        subsample = float(bp['subsample'])
                        colsample_bytree = float(bp['colsample_bytree'])
                        min_child_weight = float(bp['min_child_weight'])

                        # --- train model final (early stopping) ---
                        model_xgb = XGBRegressor(
                            n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=learning_rate_xgb, subsample=subsample,
                            colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                            objective='reg:squarederror', eval_metric='rmse',
                            random_state=seed, n_jobs=-1, early_stopping_rounds=20,
                        )
                        model_xgb.fit(
                            X_train_x, y_train_x,
                            eval_set=[(X_train_x, y_train_x), (X_val_x, y_val_x)],
                            verbose=False
                        )

                        # --- evaluasi test set (konversi balik ke skala harga) ---
                        def return_to_price(return_scaled_arr, start_price):
                            returns = scaler_return.inverse_transform(return_scaled_arr.reshape(-1, 1)).flatten()
                            return start_price * np.exp(np.cumsum(returns))

                        rmse_xgb = mae_xgb = mape_xgb = None
                        if len(X_test_x) > 0:
                            y_pred_test_scaled = model_xgb.predict(X_test_x)
                            test_start_idx = target_idx[test_mask][0]
                            start_price_test = df["Close"].iloc[test_start_idx - 1]
                            pred_prices = return_to_price(y_pred_test_scaled, start_price_test)
                            actual_prices = return_to_price(y_test_x, start_price_test)

                            rmse_xgb = np.sqrt(mean_squared_error(actual_prices, pred_prices))
                            mae_xgb = mean_absolute_error(actual_prices, pred_prices)
                            mape_xgb = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100

                        # --- forecast ke depan (direct multi-step) ---
                        X_multi, y_multi = [], []
                        for i in range(len(log_return_scaled) - window_size_xgb - int(future_days_xgb)):
                            X_multi.append(log_return_scaled[i:i + window_size_xgb])
                            y_multi.append(log_return_scaled[i + window_size_xgb: i + window_size_xgb + int(future_days_xgb)])
                        X_multi = np.array(X_multi)
                        y_multi = np.array(y_multi)
                        target_idx_multi = np.arange(window_size_xgb, window_size_xgb + len(X_multi))
                        train_mask_multi = target_idx_multi < train_size

                        model_xgb_multi = MultiOutputRegressor(
                            XGBRegressor(
                                n_estimators=n_estimators, max_depth=max_depth,
                                learning_rate=learning_rate_xgb, subsample=subsample,
                                colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                                objective='reg:squarederror', random_state=seed,
                            ),
                            n_jobs=-1
                        )
                        model_xgb_multi.fit(X_multi[train_mask_multi], y_multi[train_mask_multi])

                        last_sequence = log_return_scaled[-window_size_xgb:].reshape(1, -1)
                        future_returns_scaled = model_xgb_multi.predict(last_sequence).flatten()
                        future_returns = scaler_return.inverse_transform(future_returns_scaled.reshape(-1, 1)).flatten()

                        last_actual_price = df["Close"].iloc[-1]
                        future_predictions_xgb = last_actual_price * np.exp(np.cumsum(future_returns))

                        last_date = df["Date"].iloc[-1]
                        future_dates_xgb = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=int(future_days_xgb))

                    st.success("Model XGBoost selesai dilatih!")

                    st.write("**Hyperparameter terbaik (Bayesian Optimization):**")
                    st.json({
                        "n_estimators": n_estimators, "max_depth": max_depth,
                        "learning_rate": round(learning_rate_xgb, 5), "subsample": round(subsample, 3),
                        "colsample_bytree": round(colsample_bytree, 3), "min_child_weight": round(min_child_weight, 3),
                    })

                    if rmse_xgb is not None:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("RMSE", f"{rmse_xgb:.4f}")
                        m2.metric("MAE", f"{mae_xgb:.4f}")
                        m3.metric("MAPE", f"{mape_xgb:.2f}%")
                        st.session_state.eval_results["XGBoost"] = {"rmse": rmse_xgb, "mae": mae_xgb, "mape": mape_xgb}
                    else:
                        st.info("Data test kosong (data terlalu sedikit) — evaluasi dilewati.")

                    st.session_state.forecast_results["XGBoost"] = {
                        "future_dates": future_dates_xgb,
                        "future_predictions": future_predictions_xgb,
                        "last_actual_price": float(df["Close"].iloc[-1]),
                        "last_actual_date": df["Date"].iloc[-1],
                    }

                    st.subheader("Hasil Prediksi - XGBoost")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df["Date"], df["Close"], label="Historical Prices")
                    ax.plot(future_dates_xgb, future_predictions_xgb, linestyle="--", color="#55A868", label="Prediction (XGBoost)")
                    ax.scatter(future_dates_xgb[-1], future_predictions_xgb[-1], color="red", zorder=5)
                    ax.text(future_dates_xgb[-1], future_predictions_xgb[-1], f"{future_predictions_xgb[-1]:.2f}")
                    ax.set_title("Prediksi Harga Saham - XGBoost (Log-Return)")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    hasil_forecast_xgb = pd.DataFrame({
                        "Date": future_dates_xgb,
                        "Predicted_Close": future_predictions_xgb
                    })
                    st.dataframe(hasil_forecast_xgb, use_container_width=True)

                    csv_forecast_xgb = hasil_forecast_xgb.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download hasil prediksi XGBoost (CSV)", csv_forecast_xgb, "prediksi_xgboost.csv", "text/csv", key="dl_xgb")

                    st.caption(
                        "⚠️ Prediksi direkonstruksi dari log-return kumulatif terhadap harga terakhir. "
                        "Gunakan hasil ini sebagai gambaran tren, bukan angka pasti."
                    )

        # =====================================================
        # PERBANDINGAN GRU vs XGBOOST (muncul kalau keduanya sudah dilatih)
        # =====================================================
        if len(st.session_state.eval_results) >= 1:
            st.divider()
            st.subheader("📊 Perbandingan Model")

            eval_df = pd.DataFrame(st.session_state.eval_results).T
            eval_df.columns = ["RMSE", "MAE", "MAPE (%)"]
            st.dataframe(eval_df.round(4), use_container_width=True)

            if len(st.session_state.forecast_results) == 2:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df["Date"], df["Close"], label="Historical Prices", color="black")
                for nama, warna in [("GRU", "#DD8452"), ("XGBoost", "#55A868")]:
                    if nama in st.session_state.forecast_results:
                        fr = st.session_state.forecast_results[nama]
                        ax.plot(fr["future_dates"], fr["future_predictions"], linestyle="--", color=warna, label=f"Prediction ({nama})")
                ax.set_title("Perbandingan Prediksi GRU vs XGBoost")
                ax.set_xlabel("Date"); ax.set_ylabel("Price")
                ax.legend(); ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)


# =========================================================
# HALAMAN 4 — RINGKASAN & REKOMENDASI KPI
# =========================================================
with tab4:
    st.title("Ringkasan & Rekomendasi KPI")

    company_data = st.session_state.df_keuangan
    if not company_data:
        st.warning("Silakan upload data fundamental dulu di halaman '1. Upload Data'.")
    else:
        perusahaan_dipilih = st.selectbox("Pilih perusahaan", sorted(company_data.keys()))
        pdf = company_data[perusahaan_dipilih].sort_values("Waktu")
        data_terbaru = pdf.iloc[-1]

        st.caption(f"Data terbaru: periode **{data_terbaru['Waktu']}**")

        # =====================================================
        # A. FINANCIAL METRICS (kartu ringkasan)
        # =====================================================
        st.subheader("Financial Metrics")
        st.divider()

        def ambil(kol):
            return data_terbaru[kol] if kol in pdf.columns and pd.notna(data_terbaru[kol]) else None

        cr, der, roa, roe, eps, per = (
            ambil("CR"), ambil("DER"), ambil("ROA"), ambil("ROE"), ambil("EPS"), ambil("PER")
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Ratio (CR)", f"{cr:.2f}" if cr is not None else "Data tidak tersedia")
        c2.metric("Debt to Equity (DER)", f"{der:.2f}" if der is not None else "Data tidak tersedia")
        c3.metric("Price to Earnings (PER)", f"{per:.2f}" if per is not None else "Data tidak tersedia")

        c4, c5, c6 = st.columns(3)
        c4.metric("ROE", f"{roe:.2f}%" if roe is not None else "Data tidak tersedia")
        c5.metric("ROA", f"{roa:.2f}%" if roa is not None else "Data tidak tersedia")
        c6.metric("EPS", f"Rp {eps:,.2f}" if eps is not None else "Data tidak tersedia")

        # =====================================================
        # B. REKOMENDASI DAN KPI
        # =====================================================
        st.subheader("Rekomendasi dan KPI")
        st.divider()

        st.markdown("#### Financial Health Indicators")
        st.caption("Ambang batas berikut adalah aturan umum (rule of thumb), bisa kamu sesuaikan sendiri.")

        with st.expander("⚙️ Sesuaikan ambang batas (opsional)"):
            batas_cr  = st.number_input("CR sehat jika ≥", value=1.0, step=0.1)
            batas_der = st.number_input("DER aman jika ≤", value=1.0, step=0.1)
            batas_roa = st.number_input("ROA baik jika ≥ (%)", value=1.5, step=0.1)
            batas_roe = st.number_input("ROE tinggi jika ≥ (%)", value=10.0, step=0.5)
            batas_per = st.number_input("PER wajar jika ≤", value=20.0, step=1.0)

        st.markdown("#### Potensi Saham Berdasarkan Metrik Keuangan")

        kriteria = []
        if cr is not None:
            kriteria.append(("Current Ratio sehat (≥ %.1f)" % batas_cr, cr >= batas_cr))
        if der is not None:
            kriteria.append(("Debt to Equity aman (≤ %.1f)" % batas_der, der <= batas_der))
        if roa is not None:
            kriteria.append(("ROA baik (≥ %.1f%%)" % batas_roa, roa >= batas_roa))
        if roe is not None:
            kriteria.append(("ROE tinggi (≥ %.1f%%)" % batas_roe, roe >= batas_roe))
        if per is not None:
            kriteria.append(("PER wajar (≤ %.1f)" % batas_per, 0 < per <= batas_per))

        if not kriteria:
            st.info("Data belum cukup untuk menghitung indikator.")
        else:
            cols = st.columns(len(kriteria))
            for col, (label, lolos) in zip(cols, kriteria):
                if lolos:
                    col.success(label)
                else:
                    col.error(label)

        st.divider()

        # =====================================================
        # C. KPI ANALYSIS (hasil prediksi model)
        # =====================================================
        st.markdown("#### KPI Analysis — Hasil Prediksi Harga Saham")

        model_tersedia = list(st.session_state.forecast_results.keys())

        if not model_tersedia:
            st.info(
                "Belum ada hasil prediksi. Silakan latih model (GRU dan/atau XGBoost) dulu "
                "di halaman '3. Prediksi Harga (GRU & XGBoost)', lalu kembali ke sini."
            )
        else:
            model_dipilih = st.selectbox("Pilih model untuk ditampilkan", model_tersedia)
            forecast = st.session_state.forecast_results[model_dipilih]
            warna_model = "#DD8452" if model_dipilih == "GRU" else "#55A868"

            harga_awal = forecast["last_actual_price"]
            harga_akhir = float(forecast["future_predictions"][-1])
            tanggal_akhir = forecast["future_dates"][-1]
            perubahan_persen = (harga_akhir - harga_awal) / harga_awal * 100

            st.metric(
                label=f"Prediksi harga pada {pd.Timestamp(tanggal_akhir).strftime('%d %b %Y')} ({model_dipilih})",
                value=f"Rp {harga_akhir:,.2f}",
                delta=f"{perubahan_persen:+.2f}% dari harga terakhir (Rp {harga_awal:,.2f})",
            )

            if perubahan_persen > 5:
                st.success(
                    f"📈 Model {model_dipilih} memprediksi tren **KENAIKAN** harga sebesar "
                    f"{perubahan_persen:.2f}% dalam {len(forecast['future_dates'])} hari kerja ke depan "
                    f"(dari Rp {harga_awal:,.2f} menjadi Rp {harga_akhir:,.2f})."
                )
            elif perubahan_persen < -5:
                st.error(
                    f"📉 Model {model_dipilih} memprediksi tren **PENURUNAN** harga sebesar "
                    f"{abs(perubahan_persen):.2f}% dalam {len(forecast['future_dates'])} hari kerja ke depan "
                    f"(dari Rp {harga_awal:,.2f} menjadi Rp {harga_akhir:,.2f})."
                )
            else:
                st.warning(
                    f"➡️ Model {model_dipilih} memprediksi harga relatif **STABIL** ({perubahan_persen:+.2f}%) "
                    f"dalam {len(forecast['future_dates'])} hari kerja ke depan "
                    f"(dari Rp {harga_awal:,.2f} menjadi Rp {harga_akhir:,.2f})."
                )

            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.plot(forecast["future_dates"], forecast["future_predictions"], marker="o", markersize=3, linestyle="--", color=warna_model)
            ax.set_title(f"Ringkasan Tren Prediksi - {model_dipilih}")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Harga Prediksi")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            if model_dipilih in st.session_state.eval_results:
                ev = st.session_state.eval_results[model_dipilih]
                e1, e2, e3 = st.columns(3)
                e1.metric("RMSE (test set)", f"{ev['rmse']:.4f}")
                e2.metric("MAE (test set)", f"{ev['mae']:.4f}")
                e3.metric("MAPE (test set)", f"{ev['mape']:.2f}%")

            st.caption(
                f"⚠️ Ini hasil model {model_dipilih}, bukan saran investasi. "
                "Prediksi jangka panjang cenderung kurang akurat karena error terakumulasi "
                "setiap langkah — gunakan sebagai gambaran tren, bukan angka pasti."
            )

        # =====================================================
        # D. Tren historis ringkas (opsional, biar tidak cuma 1 titik data)
        # =====================================================
        st.divider()
        st.markdown("#### Tren Historis")
        kolom_rasio_ada = [c for c in ["CR", "DER", "ROA", "ROE", "EPS", "PER"] if c in pdf.columns]
        if kolom_rasio_ada:
            pilih_rasio_tren = st.selectbox("Pilih rasio untuk dilihat trennya", kolom_rasio_ada)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(pdf["Waktu"].astype(str), pdf[pilih_rasio_tren], marker="o")
            ax.set_title(f"Tren {pilih_rasio_tren} — {perusahaan_dipilih}")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
