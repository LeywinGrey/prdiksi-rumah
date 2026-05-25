import streamlit as st

import pandas as pd
import numpy as np

import pickle

# =========================
# LOAD MODEL
# =========================
model = pickle.load(
    open('model_rumah.pkl', 'rb')
)

scaler = pickle.load(
    open('scaler.pkl', 'rb')
)

# =========================
# TITLE
# =========================
st.title("Prediksi Harga Rumah Wilayah Jakarta")

st.write(
    "Masukkan data rumah untuk memprediksi harga."
)

# =========================
# INPUT USER
# =========================
LB = st.number_input(
    "Luas Bangunan (m²)",
    min_value=1
)

LT = st.number_input(
    "Luas Tanah (m²)",
    min_value=1
)

KT = st.number_input(
    "Kamar Tidur",
    min_value=1
)

KM = st.number_input(
    "Kamar Mandi",
    min_value=1
)

GRS = st.number_input(
    "Garasi",
    min_value=0
)

# INPUT HARGA
HARGA = st.number_input(
    "Harga Rumah (Rp)",
    min_value=100_000_000,
    value=1_000_000_000,
    step=50_000_000,
    help="Masukkan harga rumah dalam Rupiah. HARGA_PER_LT akan dihitung otomatis dari Harga / Luas Tanah."
)

# Hitung dan tampilkan HARGA_PER_LT otomatis
if LT > 0:
    HARGA_PER_LT = HARGA / LT
    if HARGA_PER_LT >= 1_000_000:
        label_per_lt = f"Rp {HARGA_PER_LT/1_000_000:.2f} Juta/m²"
    else:
        label_per_lt = f"Rp {HARGA_PER_LT:,.0f}/m²"
    st.info(f"📐 Harga per Meter Tanah (otomatis): **{label_per_lt}**")
else:
    HARGA_PER_LT = 0

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
    rumah_scaled = scaler.transform(
        rumah_baru
    )

    # predict
    pred_log = model.predict(
        rumah_scaled
    )

    # inverse log
    harga = np.expm1(
        pred_log
    )[0]

    # format harga
    if harga >= 1_000_000_000:

        hasil = (
            f"Rp {harga/1_000_000_000:.2f} Miliar"
        )

    else:

        hasil = (
            f"Rp {harga/1_000_000:.2f} Juta"
        )

    # output
    st.success(
        f"Prediksi Harga Rumah: {hasil}"
    )