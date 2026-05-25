import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open('model_rumah.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

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
# PILIHAN MODE HARGA_PER_LT
# =========================
st.divider()
st.subheader("Harga per Meter Tanah")

mode_harga = st.radio(
    "Pilih cara menentukan harga per meter tanah:",
    options=["Hitung otomatis dari Harga Rumah", "Masukkan manual"],
    index=0,
    horizontal=True
)

if mode_harga == "Hitung otomatis dari Harga Rumah":
    HARGA = st.number_input(
        "Harga Rumah (Rp)",
        min_value=100_000_000,
        value=1_000_000_000,
        step=50_000_000,
        help="HARGA_PER_LT akan dihitung otomatis: Harga / Luas Tanah"
    )
    if LT > 0:
        HARGA_PER_LT = HARGA / LT
        if HARGA_PER_LT >= 1_000_000:
            label_per_lt = f"Rp {HARGA_PER_LT/1_000_000:.2f} Juta/m²"
        else:
            label_per_lt = f"Rp {HARGA_PER_LT:,.0f}/m²"
        st.info(f"📐 Harga per Meter Tanah (otomatis): **{label_per_lt}**")
    else:
        HARGA_PER_LT = 0

else:  # Masukkan manual
    HARGA_PER_LT = st.number_input(
        "Harga per Meter Tanah (Rp/m²)",
        min_value=1_000_000,
        value=5_000_000,
        step=500_000,
        help="Masukkan harga per meter tanah secara langsung"
    )
    if HARGA_PER_LT >= 1_000_000:
        label_per_lt = f"Rp {HARGA_PER_LT/1_000_000:.2f} Juta/m²"
    else:
        label_per_lt = f"Rp {HARGA_PER_LT:,.0f}/m²"
    st.info(f"📐 Harga per Meter Tanah (manual): **{label_per_lt}**")

st.divider()

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

    # scaling
    rumah_scaled = scaler.transform(rumah_baru)

    # predict
    pred_log = model.predict(rumah_scaled)

    # inverse log
    harga = np.expm1(pred_log)[0]

    # format harga
    if harga >= 1_000_000_000:
        hasil = f"Rp {harga/1_000_000_000:.2f} Miliar"
    else:
        hasil = f"Rp {harga/1_000_000:.2f} Juta"

    st.success(f"Prediksi Harga Rumah: {hasil}")