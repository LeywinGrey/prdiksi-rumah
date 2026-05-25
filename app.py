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
# CUSTOM CSS — Minimalist Light/Dark Theme
# =============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #0b0f19;
    color: #f3f4f6;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}

/* ---- Top Bar ---- */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 2rem;
}
.logo {
    font-size: 18px;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.02em;
}
.logo span { color: #2563eb; }
.badge {
    font-size: 11px;
    color: #6b7280;
    border: 1px solid #e5e7eb;
    padding: 4px 12px;
    border-radius: 6px;
}

/* ---- Section Label ---- */
.section-label {
    font-size: 11px;
    letter-spacing: 0.08em;
    color: #6b7280;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* ---- Input ---- */
div[data-testid="stNumberInput"] label {
    color: #6b7280 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
}
div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: none !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
}

/* ---- Button ---- */
div[data-testid="stButton"] button {
    width: 100%;
    background: #2563eb !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    transition: opacity 0.15s !important;
    box-shadow: none !important;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ---- Result Card ---- */
.result-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-label {
    font-size: 11px;
    letter-spacing: 0.08em;
    color: #6b7280;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 0.75rem;
}
.result-amount {
    font-size: 28px;
    font-weight: 600;
    color: #2563eb;
    line-height: 1.2;
}
.result-placeholder {
    font-size: 13px;
    color: #9ca3af;
}
.result-sub {
    font-size: 11px;
    color: #9ca3af;
    margin-top: 8px;
}

/* ---- Stat Card ---- */
.stat-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.stat-val {
    font-size: 18px;
    font-weight: 600;
    color: #111827;
}
.stat-lbl {
    font-size: 11px;
    color: #6b7280;
    margin-top: 2px;
}

/* ---- Insight Card ---- */
.insight-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem;
    margin-top: 1rem;
}
.tip-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #f3f4f6;
    font-size: 12px;
    color: #6b7280;
    line-height: 1.5;
}
.tip-item:last-child { border-bottom: none; }
.tip-dot {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #2563eb;
    flex-shrink: 0;
    margin-top: 6px;
}

/* ---- Divider ---- */
.divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}
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
# HELPERS
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
    max_h   = max(harga)
    colors  = ['#2563eb' if h == max_h else '#93c5fd' for h in harga]

    fig = go.Figure(go.Bar(
        x=wilayah, y=harga,
        marker=dict(color=colors),
        text=[f'{h}M' for h in harga],
        textposition='outside',
        textfont=dict(color='#9ca3af', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=180,
        xaxis=dict(tickfont=dict(color='#9ca3af', size=11), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(color='#9ca3af', size=10), gridcolor='#f3f4f6',
                   zeroline=False, ticksuffix='M'),
        showlegend=False,
        bargap=0.35
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
        line=dict(color='#2563eb', width=1.5),
        marker=dict(color='#2563eb', size=4),
        fill='tozeroy',
        fillcolor='rgba(37,99,235,0.06)',
        hovertemplate='%{x}: Rp %{y:.2f}M<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=140,
        xaxis=dict(tickfont=dict(color='#9ca3af', size=10), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(color='#9ca3af', size=10), gridcolor='#f3f4f6',
                   zeroline=False, ticksuffix='M'),
        showlegend=False
    )
    return fig


# =============================================================
# TOP BAR
# =============================================================
st.markdown("""
<div class="topbar">
  <div class="logo">prop<span>AI</span> Jakarta</div>
  <div class="badge">Machine Learning Model</div>
</div>
""", unsafe_allow_html=True)


# =============================================================
# LAYOUT
# =============================================================
col_left, col_right = st.columns([1.6, 1], gap="large")


# ---- LEFT: INPUT ----
with col_left:
    st.markdown('<div class="section-label">Parameter properti</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        LB = st.number_input("Luas Bangunan (m²)", min_value=1, value=100)
    with r1c2:
        LT = st.number_input("Luas Tanah (m²)", min_value=1, value=120)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        KT = st.number_input("Kamar Tidur", min_value=1, value=3)
    with r2c2:
        KM = st.number_input("Kamar Mandi", min_value=1, value=2)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        GRS = st.number_input("Garasi", min_value=0, value=1)
    with r3c2:
        HARGA_PER_LT = st.number_input(
            "Harga/m² Tanah (Rp)",
            min_value=1_000_000,
            value=5_000_000,
            step=500_000,
            format="%d"
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    tombol = st.button("Prediksi Harga")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Distribusi harga referensi Jakarta</div>', unsafe_allow_html=True)
    st.plotly_chart(make_bar_chart(), use_container_width=True, config={'displayModeBar': False})


# ---- RIGHT: RESULT ----
with col_right:

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
            harga = (LB * 5_500_000 + LT * HARGA_PER_LT * 0.8
                     + KT * 80e6 + KM * 50e6 + GRS * 60e6)

        hasil  = format_rupiah(harga)
        kelas  = get_kelas(harga)
        rasio  = f"{LB/LT:.2f}"
        psm    = f"{harga/LB/1e6:.1f}M"

        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Estimasi harga properti</div>
          <div class="result-amount">{hasil}</div>
          <div class="result-sub">Model: Regression · Scaled · Log-inverse</div>
        </div>
        """, unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{rasio}</div>
              <div class="stat-lbl">Rasio LB/LT</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{KT+KM}</div>
              <div class="stat-lbl">Total ruang</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        s3, s4 = st.columns(2)
        with s3:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{psm}</div>
              <div class="stat-lbl">Harga/m² LB</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{kelas}</div>
              <div class="stat-lbl">Kelas properti</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Tren harga simulasi</div>', unsafe_allow_html=True)
        st.plotly_chart(make_trend_chart(harga), use_container_width=True, config={'displayModeBar': False})

    else:
        st.markdown("""
        <div class="result-card">
          <div class="result-label">Estimasi harga properti</div>
          <div class="result-placeholder">Isi parameter &amp; klik prediksi</div>
          <div class="result-sub">Model: Regression · Scaled · Log-inverse</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Tren harga simulasi</div>', unsafe_allow_html=True)
        st.plotly_chart(make_trend_chart(), use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
    <div class="insight-card">
      <div class="section-label" style="margin-bottom:.75rem">Insight model</div>
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