import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open('model_rumah.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Median HARGA_PER_LT dari data training (RobustScaler center)
# Sesuai notebook: HARGA_PER_LT = HARGA / LT
DEFAULT_HARGA_PER_LT = 29_629_629

# =========================
# TITLE
# =========================
st.title("Prediksi Harga Rumah Wilayah Jakarta")
st.write("Masukkan data rumah untuk memprediksi harga.")

# =========================
# INPUT USER
# =========================
LB = st.number_input("Luas Bangunan (m²)", min_value=1)
LT = st.number_input("Luas Tanah (m²)", min_value=1)
KT = st.number_input("Kamar Tidur", min_value=1)
KM = st.number_input("Kamar Mandi", min_value=1)
GRS = st.number_input("Garasi", min_value=0)

# =========================
# FORMAT RUPIAH
# =========================
def format_rupiah(angka):
    if angka >= 1_000_000_000:
        return f"Rp {angka/1_000_000_000:.2f} miliar"
    elif angka >= 1_000_000:
        juta = int(angka) // 1_000_000
        sisa = int(angka) % 1_000_000 // 1_000
        if sisa > 0:
            return f"Rp {juta}.{sisa:03d}.000"
        else:
            return f"Rp {juta}.000.000"
    else:
        return f"Rp {int(angka):,}".replace(",", ".")

# =========================
# HARGA TOTAL RUMAH (OPSIONAL)
# =========================
with st.expander("➕ Masukkan Harga Rumah (opsional)"):
    st.caption(
        "Harga rumah digunakan untuk menghitung harga per meter tanah "
        "(Harga Rumah ÷ Luas Tanah), sesuai data training. "
        "Jika tidak diisi, akan menggunakan nilai median dari data training "
        "yaitu Rp 29.629.629 per meter persegi."
    )
    gunakan_harga_manual = st.checkbox("Saya ingin memasukkan harga sendiri")

    if gunakan_harga_manual:
        HARGA = st.number_input(
            "Harga Rumah (Rp)",
            min_value=0,
            value=0,
            step=1_000_000,
            format="%d",
            help="Contoh: 600000000 untuk Rp 600 juta"
        )

        HARGA_PER_LT = HARGA / LT if LT > 0 else DEFAULT_HARGA_PER_LT

        if HARGA > 0:
            st.info(
                f"📐 Harga per Meter Tanah: **{format_rupiah(HARGA_PER_LT)} per m²** "
                f"({format_rupiah(HARGA)} ÷ {LT} m²)"
            )
    else:
        HARGA_PER_LT = DEFAULT_HARGA_PER_LT

# =========================
# BUTTON
# =========================
if st.button("Prediksi"):

    TOTAL_RUANG = KT + KM

    rumah_baru = pd.DataFrame({
        'LB': [LB],
        'LT': [LT],
        'KT': [KT],
        'KM': [KM],
        'GRS': [GRS],
        'TOTAL_RUANG': [TOTAL_RUANG],
        'HARGA_PER_LT': [HARGA_PER_LT]
    })

    rumah_scaled = scaler.transform(rumah_baru)
    pred_log = model.predict(rumah_scaled)
    harga = np.expm1(pred_log)[0]

    if harga >= 1_000_000_000:
        hasil = f"Rp {harga/1_000_000_000:.2f} miliar"
    else:
        hasil = f"Rp {harga/1_000_000:.2f} juta"

    st.success(f"Prediksi Harga Rumah: {hasil}")