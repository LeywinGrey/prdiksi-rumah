import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="propAI Jakarta",
    page_icon="🏠",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open('model_rumah.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.stApp {
    background-color: #0a0d12;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    color: #94a3b8;
    margin-bottom: 30px;
}

.card {
    background: #111827;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}

.result-card {
    background: linear-gradient(135deg,#0c1628,#0f1e35);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid rgba(56,189,248,0.2);
}

.big-price {
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
}

.stButton>button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    background: linear-gradient(135deg,#0369a1,#0ea5e9);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="title">🏠 propAI Jakarta</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Prediksi Harga Rumah Menggunakan Machine Learning</div>',
    unsafe_allow_html=True
)

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([2,1])

# =========================
# LEFT SIDE
# =========================
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Parameter Properti")

    c1, c2 = st.columns(2)

    with c1:
        LB = st.number_input("Luas Bangunan (m²)", min_value=1)

    with c2:
        LT = st.number_input("Luas Tanah (m²)", min_value=1)

    c3, c4 = st.columns(2)

    with c3:
        KT = st.number_input("Kamar Tidur", min_value=1)

    with c4:
        KM = st.number_input("Kamar Mandi", min_value=1)

    c5, c6 = st.columns(2)

    with c5:
        GRS = st.number_input("Garasi", min_value=0)

    with c6:
        HARGA_PER_LT = st.number_input(
            "Harga/m² Tanah",
            min_value=1000000,
            value=5000000,
            step=500000
        )

    prediksi = st.button("Prediksi Harga")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# RIGHT SIDE
# =========================
with col2:

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.markdown("### Estimasi Harga")

    if prediksi:

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

        st.markdown(
            f'<div class="big-price">{hasil}</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="big-price">Menunggu Prediksi</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)