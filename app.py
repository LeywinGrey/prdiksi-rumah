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
# HARGA PER METER TANAH (OPSIONAL)
# =========================
with st.expander("➕ Masukkan Harga per Meter Tanah (opsional)"):
    st.caption(
        "Jika tidak diisi, harga per meter tanah akan menggunakan "
        "nilai median dari data training "
        f"(Rp {DEFAULT_HARGA_PER_LT/1_000_000:.1f} Juta/m²)."
    )
    gunakan_harga_manual = st.checkbox("Saya ingin memasukkan harga sendiri")

    if gunakan_harga_manual:
        HARGA = st.number_input(
            "Harga Rumah (Rp)",
            min_value=100_000_000,
            value=1_000_000_000,
            step=50_000_000,
        )
        HARGA_PER_LT = HARGA / LT if LT > 0 else DEFAULT_HARGA_PER_LT
        label = f"Rp {HARGA_PER_LT/1_000_000:.2f} Juta/m²"
        st.info(f"📐 Harga per Meter Tanah: **{label}**")
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
        hasil = f"Rp {harga/1_000_000_000:.2f} Miliar"
    else:
        hasil = f"Rp {harga/1_000_000:.2f} Juta"

    st.success(f"Prediksi Harga Rumah: {hasil}")