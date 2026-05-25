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

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #09090b;
    color: #e5e7eb;
}

#MainMenu,
footer,
header {
    visibility: hidden;
}

.block-container {
    padding-top: 0rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

/* HERO */

.hero-bar {
    background: #111111;
    border-bottom: 1px solid #1f2937;
    padding: 18px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
}

.hero-logo {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: white;
}

.hero-logo span {
    color: #3b82f6;
}

.hero-badge {
    background: #111827;
    color: #60a5fa;
    border-radius: 30px;
    padding: 6px 14px;
    font-size: 11px;
    border: 1px solid #1e293b;
}

/* SECTION TITLE */

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #60a5fa;
    text-transform: uppercase;
    margin-bottom: 18px;
}

/* INPUT */

div[data-testid="stNumberInput"] label {
    color: #9ca3af !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

div[data-testid="stNumberInput"] input {
    background: #111827 !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    padding: 14px !important;
    font-size: 15px !important;
}

div[data-testid="stNumberInput"] input:focus {
    border: 1px solid #3b82f6 !important;
    box-shadow: none !important;
}

/* BUTTON */

div[data-testid="stButton"] button {
    width: 100%;
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 15px 20px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    transition: 0.2s;
}

div[data-testid="stButton"] button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px);
}

/* RESULT CARD */

.result-card {
    background: #111827;
    border: none;
    border-radius: 24px;
    padding: 40px 30px;
    text-align: center;
    margin-bottom: 20px;
}

.result-label {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 14px;
}

.result-amount {
    font-family: 'Syne', sans-serif;
    font-size: 38px;
    font-weight: 700;
    color: white;
}

.result-placeholder {
    color: #6b7280;
    font-size: 14px;
}

.result-sub {
    color: #6b7280;
    font-size: 11px;
    margin-top: 10px;
}

/* STAT CARD */

.stat-card {
    background: #161b22;
    border: none;
    border-radius: 18px;
    padding: 22px;
    text-align: center;
}

.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    color: white;
    font-weight: 700;
}

.stat-lbl {
    color: #6b7280;
    font-size: 11px;
    margin-top: 6px;
    letter-spacing: 0.05em;
}

/* INSIGHT CARD */

.insight-card {
    background: #111827;
    border-radius: 22px;
    padding: 22px;
    margin-top: 18px;
}

.insight-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    letter-spacing: 0.12em;
    color: #9ca3af;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.tip-item {
    display: flex;
    gap: 12px;
    padding: 10px 0;
    color: #9ca3af;
    font-size: 13px;
    border-bottom: 1px solid #1f2937;
}

.tip-item:last-child {
    border-bottom: none;
}

.tip-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #3b82f6;
    margin-top: 7px;
    flex-shrink: 0;
}

/* DIVIDER */

.divider {
    border: none;
    border-top: 1px solid #1f2937;
    margin: 24px 0;
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

def get_kelas(val):

    if val < 500_000_000:
        return "Sederhana"

    elif val < 1_500_000_000:
        return "Menengah"

    elif val < 5_000_000_000:
        return "Mewah"

    return "Ultra Mewah"

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero-bar">
    <div class="hero-logo">
        prop<span>AI</span> Jakarta
    </div>

    <div class="hero-badge">
        Machine Learning Model
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT
# =========================================================
left, right = st.columns([1.5, 1])

# =========================================================
# LEFT
# =========================================================
with left:

    st.markdown(
        '<div class="section-title">Parameter Properti</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        LB = st.number_input(
            "📐 Luas Bangunan (m²)",
            min_value=1,
            value=100
        )

    with c2:
        LT = st.number_input(
            "🗺️ Luas Tanah (m²)",
            min_value=1,
            value=120
        )

    c3, c4 = st.columns(2)

    with c3:
        KT = st.number_input(
            "🛏️ Kamar Tidur",
            min_value=1,
            value=3
        )

    with c4:
        KM = st.number_input(
            "🚿 Kamar Mandi",
            min_value=1,
            value=2
        )

    c5, c6 = st.columns(2)

    with c5:
        GRS = st.number_input(
            "🚗 Garasi",
            min_value=0,
            value=1
        )

    with c6:
        HARGA_PER_LT = st.number_input(
            "💰 Harga/m² Tanah",
            min_value=1000000,
            value=5000000,
            step=500000
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    prediksi = st.button("⚡ Prediksi Harga Rumah")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Distribusi Harga Jakarta</div>',
        unsafe_allow_html=True
    )

    wilayah = ['Jak-Pus', 'Jak-Sel', 'Jak-Bar', 'Jak-Tim', 'Jak-Ut']

    harga = [3.8, 5.2, 2.9, 2.4, 2.1]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=wilayah,
        y=harga,
        marker_color='#2563eb'
    ))

    fig.update_layout(
        paper_bgcolor='#09090b',
        plot_bgcolor='#09090b',
        font_color='#9ca3af',
        height=250,
        margin=dict(l=0, r=0, t=20, b=0)
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={'displayModeBar': False}
    )

# =========================================================
# RIGHT
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

        harga_pred = np.expm1(pred_log)[0]

        hasil = format_rupiah(harga_pred)

        kelas = get_kelas(harga_pred)

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

        s1, s2 = st.columns(2)

        with s1:

            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">
                    {LB/LT:.2f}
                </div>

                <div class="stat-lbl">
                    Rasio LB/LT
                </div>
            </div>
            """, unsafe_allow_html=True)

        with s2:

            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">
                    {KT+KM}
                </div>

                <div class="stat-lbl">
                    Total Ruang
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        s3, s4 = st.columns(2)

        with s3:

            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">
                    {harga_pred/LB/1e6:.1f}M
                </div>

                <div class="stat-lbl">
                    Harga/m² LB
                </div>
            </div>
            """, unsafe_allow_html=True)

        with s4:

            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">
                    {kelas}
                </div>

                <div class="stat-lbl">
                    Kelas Properti
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card">

            <div class="result-label">
                Estimasi Harga Properti
            </div>

            <div class="result-placeholder">
                Isi parameter lalu klik prediksi
            </div>

            <div class="result-sub">
                Linear Regression Model
            </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""

    <div class="insight-card">

        <div class="insight-title">
            Insight Model
        </div>

        <div class="tip-item">
            <div class="tip-dot"></div>
            <span>
                LB dan LT menjadi fitur paling berpengaruh dalam prediksi harga rumah.
            </span>
        </div>

        <div class="tip-item">
            <div class="tip-dot"></div>
            <span>
                Harga tanah tiap wilayah Jakarta memiliki perbedaan signifikan.
            </span>
        </div>

        <div class="tip-item">
            <div class="tip-dot"></div>
            <span>
                Model menggunakan preprocessing scaling dan transformasi log.
            </span>
        </div>

    </div>

    """, unsafe_allow_html=True)