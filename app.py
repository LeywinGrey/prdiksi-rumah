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
    "Luas Bangunan",
    min_value=1
)

LT = st.number_input(
    "Luas Tanah",
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

# INPUT HARGA PER LT
HARGA_PER_LT = st.number_input(
    "Harga per Meter Tanah",
    min_value=1000000,
    value=5000000,
    step=500000
)

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