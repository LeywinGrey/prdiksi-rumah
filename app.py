import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="propAI Jakarta",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* BACKGROUND */

.stApp {
    background: #050816;
    color: #f8fafc;
}

/* HIDE STREAMLIT */

#MainMenu,
footer,
header {
    visibility: hidden;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

/* TOPBAR */

.topbar {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 3rem;

    padding-bottom: 1rem;

    border-bottom: 1px solid #111827;
}

.logo {

    font-size: 28px;

    font-weight: 700;

    color: white;

    letter-spacing: -0.03em;
}

.logo span {
    color: #3b82f6;
}

.badge {

    background: #111827;

    color: #93c5fd;

    border: 1px solid #1e293b;

    border-radius: 999px;

    padding: 8px 16px;

    font-size: 12px;
}

/* SECTION */

.section-label {

    font-size: 12px;

    font-weight: 600;

    color: #64748b;

    text-transform: uppercase;

    letter-spacing: .08em;

    margin-bottom: 1.25rem;
}

/* INPUT */

div[data-testid="stNumberInput"] label {

    color: #94a3b8 !important;

    font-size: 13px !important;

    margin-bottom: 6px !important;
}

div[data-testid="stNumberInput"] input {

    background: #0f172a !important;

    border: 1px solid #1e293b !important;

    color: white !important;

    border-radius: 14px !important;

    height: 52px !important;

    padding-left: 16px !important;

    font-size: 15px !important;
}

div[data-testid="stNumberInput"] input:focus {

    border-color: #2563eb !important;

    box-shadow: 0 0 0 1px #2563eb !important;
}

/* BUTTON */

div[data-testid="stButton"] button {

    background: #2563eb !important;

    color: white !important;

    border: none !important;

    height: 52px;

    border-radius: 14px !important;

    font-size: 15px !important;

    font-weight: 600 !important;

    width: 100%;

    transition: 0.2s;
}

div[data-testid="stButton"] button:hover {

    background: #1d4ed8 !important;

    transform: translateY(-1px);
}

/* CARD */

.card {

    background: #0f172a;

    border: 1px solid #1e293b;

    border-radius: 24px;

    padding: 24px;
}

/* RESULT CARD */

.result-card {

    background: linear-gradient(
        180deg,
        #111827,
        #0f172a
    );

    border: 1px solid #1e293b;

    border-radius: 28px;

    padding: 32px;

    text-align: center;

    margin-bottom: 18px;
}

.result-label {

    font-size: 12px;

    text-transform: uppercase;

    color: #64748b;

    letter-spacing: .08em;

    margin-bottom: 14px;
}

.result-amount {

    font-size: 42px;

    font-weight: 700;

    color: white;

    line-height: 1.1;
}

.result-sub {

    color: #64748b;

    font-size: 12px;

    margin-top: 10px;
}

/* STAT */

.stat-card {

    background: #111827;

    border: 1px solid #1e293b;

    border-radius: 18px;

    padding: 18px;

    text-align: center;
}

.stat-val {

    color: white;

    font-size: 24px;

    font-weight: 700;
}

.stat-lbl {

    color: #64748b;

    font-size: 11px;

    margin-top: 4px;
}

/* INSIGHT */

.insight-card {

    background: #0f172a;

    border: 1px solid #1e293b;

    border-radius: 24px;

    padding: 24px;

    margin-top: 18px;
}

.tip-item {

    color: #94a3b8;

    padding: 12px 0;

    border-bottom: 1px solid #1e293b;

    font-size: 14px;
}

.tip-item:last-child {
    border-bottom: none;
}

/* CHART */

.js-plotly-plot {
    border-radius: 20px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
model = pickle.load(open('model_rumah.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# =========================================================
# HELPER
# =========================================================
def format_rupiah(val):

    if val >= 1_000_000_000:
        return f"Rp {val/1_000_000_000:.2f} Miliar"

    return f"Rp {val/1_000_000:.0f} Juta"

# =========================================================
# TOPBAR
# =========================================================
st.markdown("""
<div class="topbar">
    <div class="logo">
        prop<span>AI</span> Jakarta
    </div>

    <div class="badge">
        Machine Learning Model
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT
# =========================================================
left, right = st.columns([1.5,1])

# =========================================================
# LEFT SIDE
# =========================================================
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">Parameter Properti</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        LB = st.number_input(
            "Luas Bangunan (m²)",
            min_value=1,
            value=100
        )

    with c2:
        LT = st.number_input(
            "Luas Tanah (m²)",
            min_value=1,
            value=120
        )

    c3, c4 = st.columns(2)

    with c3:
        KT = st.number_input(
            "Kamar Tidur",
            min_value=1,
            value=3
        )

    with c4:
        KM = st.number_input(
            "Kamar Mandi",
            min_value=1,
            value=2
        )

    c5, c6 = st.columns(2)

    with c5:
        GRS = st.number_input(
            "Garasi",
            min_value=0,
            value=1
        )

    with c6:
        HARGA_PER_LT = st.number_input(
            "Harga/m² Tanah",
            min_value=1000000,
            value=5000000,
            step=500000
        )

    st.markdown("<br>", unsafe_allow_html=True)

    prediksi = st.button("Prediksi Harga Rumah")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RIGHT SIDE
# =========================================================
with right:

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

        hasil = format_rupiah(harga)

        st.markdown(f"""
        <div class="result-card">

            <div class="result-label">
                Estimasi Harga Properti
            </div>

            <div class="result-amount">
                {hasil}
            </div>

            <div class="result-sub">
                Linear Regression Model
            </div>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card">

            <div class="result-label">
                Estimasi Harga Properti
            </div>

            <div class="result-amount">
                Rp 0
            </div>

            <div class="result-sub">
                Isi parameter lalu klik prediksi
            </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""

    <div class="insight-card">

        <div class="section-label">
            Insight Model
        </div>

        <div class="tip-item">
            LB dan LT menjadi fitur paling berpengaruh terhadap harga rumah.
        </div>

        <div class="tip-item">
            Harga tanah tiap wilayah Jakarta memiliki perbedaan signifikan.
        </div>

        <div class="tip-item">
            Model menggunakan preprocessing scaling dan transformasi log.
        </div>

    </div>

    """, unsafe_allow_html=True)