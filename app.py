import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="propAI Jakarta — Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================
# CUSTOM CSS — Dark Premium Theme
# =============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-color: #0d1117;
    color: #e8eaf0;
}

/* ---- Hide Streamlit default elements ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ---- Hero Bar ---- */
.hero-bar {
    background: #111827;
    border-bottom: 1px solid #1e293b;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
}
.hero-logo {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.hero-logo span { color: #0ea5e9; }
.hero-badge {
    background: #0c2233;
    border: 1px solid #164e63;
    color: #0ea5e9;
    font-size: 11px;
    font-weight: 500;
    padding: 4px 14px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}

/* ---- Section Title ---- */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #0ea5e9;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 14px;
    background: #0ea5e9;
    border-radius: 2px;
}

/* ---- Input overrides ---- */
div[data-testid="stNumberInput"] label {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="stNumberInput"] input {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-size: 15px !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #0ea5e9 !important;
    background: #0c1a27 !important;
    box-shadow: none !important;
}

/* ---- Button ---- */
div[data-testid="stButton"] button {
    width: 100%;
    background: #0ea5e9 !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    padding: 14px 20px !important;
    transition: all 0.25s !important;
}
div[data-testid="stButton"] button:hover {
    background: #0284c7 !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}

/* ---- Result card ---- */
.result-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-bottom: 16px;
}
.result-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    color: #0ea5e9;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.result-amount {
    font-family: 'Syne', sans-serif;
    font-size: 34px;
    font-weight: 700;
    color: #0ea5e9;
    line-height: 1.2;
}
.result-placeholder {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: #334155;
}
.result-sub {
    font-size: 11px;
    color: #334155;
    margin-top: 8px;
}

/* ---- Stat card ---- */
.stat-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
}
.stat-lbl {
    font-size: 11px;
    color: #475569;
    margin-top: 4px;
    letter-spacing: 0.04em;
}

/* ---- Info card ---- */
.insight-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 18px 20px;
    margin-top: 16px;
}
.tip-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #1e293b;
    font-size: 12px;
    color: #64748b;
    line-height: 1.5;
}
.tip-item:last-child { border-bottom: none; padding-bottom: 0; }
.tip-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #0ea5e9;
    flex-shrink: 0;
    margin-top: 5px;
}

/* ---- Divider ---- */
.divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 20px 0;
}

/* ---- Chart wrapper ---- */
.chart-wrap {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 18px;
    margin-top: 16px;
}

/* ---- Plotly chart transparent bg ---- */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================
# LOAD MODEL
# =============================================================
@st.cache_resource
def load_model():
    model  = pickle.load(open('model_rumah.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# =============================================================
# HELPER FUNCTIONS
# =============================================================
def format_rupiah(val: float) -> str:
    if val >= 1_000_000_000_000:
        return f"Rp {val/1_000_000_000_000:.2f} Triliun"
    if val >= 1_000_000_000:
        return f"Rp {val/1_000_000_000:.2f} Miliar"
    return f"Rp {val/1_000_000:.0f} Juta"

def get_kelas(val: float) -> str:
    if val < 500_000_000:   return "Sederhana"
    if val < 1_500_000_000: return "Menengah"
    if val < 5_000_000_000: return "Mewah"
    return "Ultra Mewah"

def make_bar_chart():
    wilayah = ['Jak-Pus', 'Jak-Sel', 'Jak-Bar', 'Jak-Tim', 'Jak-Ut']
    harga   = [3.8, 5.2, 2.9, 2.4, 2.1]
    colors  = ['#0369a1', '#0ea5e9', '#0284c7', '#075985', '#0c4a6e']
    fig = go.Figure(go.Bar(
        x=wilayah, y=harga,
        marker=dict(color=colors, line=dict(color='#0ea5e9', width=0)),
        text=[f'Rp {h}M' for h in harga],
        textposition='outside',
        textfont=dict(color='#64748b', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=200,
        xaxis=dict(tickfont=dict(color='#64748b', size=11), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(color='#475569', size=10), gridcolor='#1e293b',
                   zeroline=False, ticksuffix='M'),
        showlegend=False
    )
    return fig

def make_trend_chart(base_harga: float = None):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun']
    if base_harga:
        b = base_harga / 1e9
        data = [round(b*0.82,2), round(b*0.87,2), round(b*0.90,2),
                round(b*0.95,2), round(b*0.98,2), round(b,2)]
    else:
        data = [2.1, 2.3, 2.25, 2.4, 2.6, 2.8]

    fig = go.Figure(go.Scatter(
        x=months, y=data,
        mode='lines+markers',
        line=dict(color='#0ea5e9', width=2),
        marker=dict(color='#0ea5e9', size=5),
        fill='tozeroy',
        fillcolor='#0c2233',
        hovertemplate='%{x}: Rp %{y:.2f}M<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=150,
        xaxis=dict(tickfont=dict(color='#475569', size=10), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(color='#475569', size=10), gridcolor='#1e293b',
                   zeroline=False, ticksuffix='M'),
        showlegend=False
    )
    return fig


# =============================================================
# HERO BAR
# =============================================================
st.markdown("""
<div class="hero-bar">
  <div class="hero-logo">prop<span>AI</span> Jakarta</div>
  <div class="hero-badge">Machine Learning Model</div>
</div>
""", unsafe_allow_html=True)


# =============================================================
# LAYOUT: 2 COLUMNS
# =============================================================
col_left, col_right = st.columns([1.6, 1], gap="large")


# ===========================
# LEFT COLUMN — INPUT
# ===========================
with col_left:
    st.markdown('<div class="section-title">Parameter Properti</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        LB = st.number_input("📐 Luas Bangunan (m²)", min_value=1, value=100)
    with r1c2:
        LT = st.number_input("🗺️ Luas Tanah (m²)", min_value=1, value=120)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        KT = st.number_input("🛏️ Kamar Tidur", min_value=1, value=3)
    with r2c2:
        KM = st.number_input("🚿 Kamar Mandi", min_value=1, value=2)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        GRS = st.number_input("🚗 Garasi", min_value=0, value=1)
    with r3c2:
        HARGA_PER_LT = st.number_input(
            "💰 Harga/m² Tanah (Rp)",
            min_value=1_000_000,
            value=5_000_000,
            step=500_000,
            format="%d"
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    tombol = st.button("⚡  Prediksi Harga Sekarang")

    # Bar Chart Wilayah
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distribusi Harga Referensi Jakarta</div>', unsafe_allow_html=True)
    st.plotly_chart(make_bar_chart(), use_container_width=True, config={'displayModeBar': False})


# ===========================
# RIGHT COLUMN — RESULT
# ===========================
with col_right:

    # ---- Result Card ----
    if tombol:
        TOTAL_RUANG = KT + KM
        df_input = pd.DataFrame({
            'LB':          [LB],
            'LT':          [LT],
            'KT':          [KT],
            'KM':          [KM],
            'GRS':         [GRS],
            'TOTAL_RUANG': [TOTAL_RUANG],
            'HARGA_PER_LT':[HARGA_PER_LT]
        })

        if model_loaded:
            rumah_scaled = scaler.transform(df_input)
            pred_log     = model.predict(rumah_scaled)
            harga        = float(np.expm1(pred_log)[0])
        else:
            # Fallback simulasi jika model belum ada
            harga = (LB * 5_500_000 + LT * HARGA_PER_LT * 0.8
                     + KT * 80e6 + KM * 50e6 + GRS * 60e6)

        hasil  = format_rupiah(harga)
        kelas  = get_kelas(harga)
        rasio  = f"{LB/LT:.2f}"
        psm    = f"{harga/LB/1e6:.1f}M"

        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Estimasi Harga Properti</div>
          <div class="result-amount">{hasil}</div>
          <div class="result-sub">Model: Regression · Scaled · Log-inverse</div>
        </div>
        """, unsafe_allow_html=True)

        # Stat Cards
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{rasio}</div>
              <div class="stat-lbl">Rasio LB/LT</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{KT+KM}</div>
              <div class="stat-lbl">Total Ruang</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        s3, s4 = st.columns(2)
        with s3:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{psm}</div>
              <div class="stat-lbl">Harga/m² LB</div>
            </div>
            """, unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{kelas}</div>
              <div class="stat-lbl">Kelas Properti</div>
            </div>
            """, unsafe_allow_html=True)

        # Trend Chart
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Tren Harga Simulasi</div>', unsafe_allow_html=True)
        st.plotly_chart(make_trend_chart(harga), use_container_width=True, config={'displayModeBar': False})

    else:
        # Placeholder sebelum prediksi
        st.markdown("""
        <div class="result-card">
          <div class="result-label">Estimasi Harga Properti</div>
          <div class="result-placeholder">Isi parameter & klik prediksi</div>
          <div class="result-sub">Model: Regression · Scaled · Log-inverse</div>
        </div>
        """, unsafe_allow_html=True)

        # Trend chart default
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Tren Harga Simulasi</div>', unsafe_allow_html=True)
        st.plotly_chart(make_trend_chart(), use_container_width=True, config={'displayModeBar': False})

    # Insight Card
    st.markdown("""
    <div class="insight-card">
      <div class="section-title" style="margin-bottom:12px">Insight Model</div>
      <div class="tip-item">
        <div class="tip-dot"></div>
        <span>LB &amp; LT adalah fitur terkuat dalam model prediksi harga rumah Jakarta.</span>
      </div>
      <div class="tip-item">
        <div class="tip-dot"></div>
        <span>Harga per m² tanah bervariasi signifikan antar kecamatan di Jakarta.</span>
      </div>
      <div class="tip-item">
        <div class="tip-dot"></div>
        <span>Model dilatih dengan data scraping listing properti wilayah DKI Jakarta.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)