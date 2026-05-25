import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="propAI Jakarta", page_icon="🏠", layout="wide")

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"] {
  background: #1a1c23 !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
section[data-testid="stMain"] > div { padding: 0 !important; }

/* ── App wrapper ── */
.app-wrap {
  background: #1a1c23;
  min-height: 100vh;
  padding: 1.5rem 2rem 3rem;
  max-width: 1060px;
  margin: 0 auto;
}

/* ── Topbar ── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 1.2rem;
  border-bottom: 1px solid #2c2f3a;
  margin-bottom: 1.8rem;
}
.logo {
  font-size: 1.35rem;
  font-weight: 600;
  color: #e8eaf0;
  letter-spacing: -0.02em;
}
.logo .ai { color: #4f8ef7; }
.badge {
  background: #22253a;
  color: #8ba7e8;
  border: 1px solid #2c3555;
  border-radius: 999px;
  padding: 5px 16px;
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.02em;
}

/* ── Main grid ── */
.main-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  align-items: start;
}

/* ── Panel ── */
.panel {
  background: #22253a;
  border: 1px solid #2c2f42;
  border-radius: 14px;
  padding: 1.5rem;
}

/* ── Section label ── */
.sec-label {
  font-size: 0.65rem;
  font-weight: 700;
  color: #5a6070;
  text-transform: uppercase;
  letter-spacing: .12em;
  margin-bottom: 1.1rem;
}

/* ── Field label ── */
.f-label {
  font-size: 0.78rem;
  color: #8b90a0;
  margin-bottom: .4rem;
  display: block;
}

/* ── Field grid ── */
.field-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem 1.2rem;
  margin-bottom: 1.2rem;
}

/* ── Number input override ── */
div[data-testid="stNumberInput"] {
  margin-bottom: 0 !important;
}
div[data-testid="stNumberInput"] label {
  font-size: 0.78rem !important;
  color: #8b90a0 !important;
  font-family: 'DM Sans', sans-serif !important;
  margin-bottom: 4px !important;
}
div[data-testid="stNumberInput"] > div {
  background: #161820 !important;
  border: 1px solid #2c2f42 !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}
div[data-testid="stNumberInput"] input {
  background: transparent !important;
  color: #e8eaf0 !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
  border: none !important;
  padding: 10px 12px !important;
}
div[data-testid="stNumberInput"] button {
  background: transparent !important;
  border: none !important;
  color: #5a6070 !important;
}
div[data-testid="stNumberInput"] button:hover { color: #e8eaf0 !important; }

/* ── Chip buttons ── */
div[data-testid="stHorizontalBlock"] button {
  background: #161820 !important;
  border: 1px solid #2c2f42 !important;
  color: #8b90a0 !important;
  border-radius: 999px !important;
  font-size: 0.72rem !important;
  font-family: 'DM Sans', sans-serif !important;
  padding: 4px 10px !important;
  height: auto !important;
  min-height: 0 !important;
  font-weight: 500 !important;
  transition: all 0.15s !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
  background: #1c2b55 !important;
  color: #93c5fd !important;
  border-color: #2563eb !important;
}

/* ── Predict button ── */
div[data-testid="stButton"]:not([data-testid="stHorizontalBlock"] div[data-testid="stButton"]) button {
  width: 100% !important;
  background: #2c2f42 !important;
  color: #e8eaf0 !important;
  border: 1px solid #3a3f58 !important;
  border-radius: 10px !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  height: 48px !important;
  font-family: 'DM Sans', sans-serif !important;
  letter-spacing: -0.01em !important;
  transition: background 0.15s !important;
}
div[data-testid="stButton"]:not([data-testid="stHorizontalBlock"] div[data-testid="stButton"]) button:hover {
  background: #363b56 !important;
}

/* ── Result card ── */
.result-card {
  background: #22253a;
  border: 1px solid #2c2f42;
  border-radius: 14px;
  padding: 2rem 1.5rem 1.5rem;
  text-align: center;
  margin-bottom: 1.25rem;
}
.result-title {
  font-size: 0.65rem;
  font-weight: 700;
  color: #5a6070;
  text-transform: uppercase;
  letter-spacing: .12em;
  margin-bottom: .6rem;
}
.result-value {
  font-size: 2.4rem;
  font-weight: 700;
  color: #e8eaf0;
  letter-spacing: -0.04em;
  line-height: 1;
}
.result-sub {
  font-size: 0.7rem;
  color: #5a6070;
  margin-top: .5rem;
}

/* ── Stats row ── */
.stats-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: .75rem;
  margin-bottom: 1.25rem;
}
.stat-box {
  background: #161820;
  border: 1px solid #2c2f42;
  border-radius: 10px;
  padding: 1rem;
  text-align: center;
}
.stat-val { font-size: 1rem; font-weight: 700; color: #e8eaf0; }
.stat-lbl { font-size: 0.65rem; color: #5a6070; margin-top: 3px; }

/* ── Insight card ── */
.insight-card {
  background: #22253a;
  border: 1px solid #2c2f42;
  border-radius: 14px;
  padding: 1.3rem 1.4rem;
}
.insight-item {
  display: flex;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid #2c2f42;
  font-size: 0.78rem;
  color: #8b90a0;
  align-items: flex-start;
  line-height: 1.5;
}
.insight-item:last-child { border-bottom: none; padding-bottom: 0; }
.i-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: #4f8ef7;
  margin-top: 5px;
  flex-shrink: 0;
}

/* ── Bottom charts ── */
.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-top: 1.5rem;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── MODEL ───────────────────────────────────────────────────────────────────────
CENTER = np.array([210, 161, 4, 3, 2, 8, 29629629.6])
SCALE  = np.array([200, 154.25, 1, 1, 1, 3, 11973304.5])
COEF   = np.array([-0.01692395, 0.60847173, 0.0253252, 0.01994662,
                    0.02336471, 0.01509061, 0.38533976])
INTERCEPT = 22.12362834422056

def predict_price(lb, lt, kt, km, grs, hlt):
    tr  = kt + km
    raw = np.array([lb, lt, kt, km, grs, tr, hlt], dtype=float)
    lp  = INTERCEPT + np.sum(COEF * (raw - CENTER) / SCALE)
    return np.expm1(lp)

def fmt_rp(v):
    if v >= 1e9:
        return f"Rp {v/1e9:.2f} M"
    return f"Rp {v/1e6:,.0f} Jt"

def fmt_short(v):
    if v >= 1e9: return f"{v/1e9:.2f} M"
    return f"{v/1e6:,.0f} Jt"

# ── CHART THEME ─────────────────────────────────────────────────────────────────
BG   = "#22253a"
GRID = "rgba(255,255,255,0.05)"
TICK = "rgba(200,205,220,0.45)"

# ── HEADER ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-wrap" id="top">
  <div class="topbar">
    <div class="logo">prop<span class="ai">AI</span> Jakarta</div>
    <div class="badge">Machine Learning Model</div>
  </div>
""", unsafe_allow_html=True)

# ── MAIN LAYOUT ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ───────────────── LEFT PANEL ──────────────────────────────────────────────────
with col_left:
    st.markdown("<p class='sec-label' style='font-size:.65rem;font-weight:700;color:#5a6070;text-transform:uppercase;letter-spacing:.12em;margin-bottom:1rem'>Parameter Properti</p>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        lb  = st.number_input("Luas bangunan (m²)", min_value=0, value=100, step=10)
        kt  = st.number_input("Kamar tidur",         min_value=0, value=3,   step=1)
        grs = st.number_input("Garasi",               min_value=0, value=1,   step=1)
    with c2:
        lt  = st.number_input("Luas tanah (m²)",     min_value=0, value=120, step=10)
        km  = st.number_input("Kamar mandi",          min_value=0, value=2,   step=1)

    st.markdown("<p style='font-size:.78rem;color:#8b90a0;margin:.8rem 0 .4rem'>Harga/m² tanah (Rp)</p>",
                unsafe_allow_html=True)

    # Chip buttons
    PRESETS = {"15 jt": 15_000_000, "30 jt (median)": 29_629_630,
               "40 jt": 40_000_000, "60 jt": 60_000_000}
    chip_cols = st.columns(4, gap="small")
    for i, (lbl, val) in enumerate(PRESETS.items()):
        with chip_cols[i]:
            if st.button(lbl, key=f"chip_{i}"):
                st.session_state["hlt"] = val

    hlt = st.number_input("", min_value=0,
                          value=st.session_state.get("hlt", 5_000_000),
                          step=1_000_000, label_visibility="collapsed")

    st.markdown("<div style='margin-top:.75rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Prediksi harga", key="predict_main")

    # ── Bar chart (bottom-left) ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:.65rem;font-weight:700;color:#5a6070;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.6rem'>Distribusi Harga Referensi Jakarta</p>",
                unsafe_allow_html=True)

    wilayah = ["Jak-Pus","Jak-Sel","Jak-Bar","Jak-Tim","Jak-Ut"]
    median_h = [3800, 5200, 2900, 2400, 2100]

    fig_bar = go.Figure(go.Bar(
        x=median_h, y=wilayah, orientation="h",
        marker_color=["#4f8ef7","#3b7de8","#6aa3f9","#8dbeff","#b0d4ff"],
        text=[f"{v/1000:.1f}M" if v>=1000 else f"{v}Jt" for v in median_h],
        textposition="outside",
        textfont=dict(color=TICK, size=10),
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=0, b=0), height=160,
        xaxis=dict(tickfont=dict(color=TICK, size=10), gridcolor=GRID,
                   showgrid=True, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(color=TICK, size=11), gridcolor=GRID,
                   showgrid=False, categoryorder="array",
                   categoryarray=list(reversed(wilayah))),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ───────────────── RIGHT PANEL ─────────────────────────────────────────────────
with col_right:
    # Prediction state
    if predict_btn:
        st.session_state["has_pred"] = True
        st.session_state["price"] = predict_price(lb, lt, kt, km, grs, hlt)
        st.session_state["lb"] = lb
        st.session_state["lt"] = lt
        st.session_state["kt"] = kt
        st.session_state["km"] = km
        st.session_state["grs"] = grs

    if st.session_state.get("has_pred"):
        price = st.session_state["price"]
        p_lb  = st.session_state.get("lb", lb)
        p_lt  = st.session_state.get("lt", lt)
        price_str = fmt_rp(price)
        s1 = fmt_short(price / p_lb) if p_lb > 0 else "—"
        s2 = fmt_short(price / p_lt) if p_lt > 0 else "—"
        sub = "Model: Regression · Scaled · Log-inverse"
    else:
        price = None
        price_str = "—"
        s1 = s2 = "—"
        sub = "Model: Regression · Scaled · Log-inverse"

    # Result card
    st.markdown(f"""
    <div class="result-card">
      <div class="result-title">Estimasi Harga Properti</div>
      <div class="result-value">{price_str}</div>
      <div class="result-sub">{"Isi parameter &amp; klik prediksi" if not st.session_state.get("has_pred") else sub}</div>
      <div class="result-sub" style="margin-top:.3rem">{sub if not st.session_state.get("has_pred") else ""}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tren Harga Simulasi (line chart) ──
    st.markdown("<p style='font-size:.65rem;font-weight:700;color:#5a6070;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.5rem'>Tren Harga Simulasi</p>",
                unsafe_allow_html=True)

    months = ["Jan","Feb","Mar","Apr","Mei","Jun"]
    _lb  = st.session_state.get("lb", 100) or 100
    _lt  = st.session_state.get("lt", 120) or 120
    _kt  = st.session_state.get("kt", kt)
    _km  = st.session_state.get("km", km)
    _grs = st.session_state.get("grs", grs)
    base_hlt = hlt if hlt > 0 else 5_000_000
    trend_prices = [
        predict_price(_lb, _lt, _kt, _km, _grs, base_hlt * (1 + 0.04*i)) / 1e6
        for i in range(6)
    ]

    fig_line = go.Figure(go.Scatter(
        x=months, y=trend_prices,
        mode="lines+markers",
        line=dict(color="#4f8ef7", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.12)",
        marker=dict(color="#4f8ef7", size=5),
    ))
    fig_line.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=5, b=0), height=140,
        xaxis=dict(tickfont=dict(color=TICK, size=11), showgrid=False,
                   zeroline=False, linecolor="#2c2f42"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        showlegend=False, hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

    # Stats
    st.markdown(f"""
    <div class="stats-row" style="display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin:.75rem 0">
      <div class="stat-box">
        <div class="stat-val">{s1}</div>
        <div class="stat-lbl">Rp/m² bangunan</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">{s2}</div>
        <div class="stat-lbl">Rp/m² tanah</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Insight card
    st.markdown("""
    <div class="insight-card">
      <div class="sec-label" style="font-size:.65rem;font-weight:700;color:#5a6070;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.75rem">Insight Model</div>
      <div class="insight-item">
        <div class="i-dot"></div>
        <span>LB &amp; LT adalah fitur terkuat dalam model prediksi harga rumah Jakarta.</span>
      </div>
      <div class="insight-item">
        <div class="i-dot"></div>
        <span>Harga per m² tanah bervariasi signifikan antar kecamatan di Jakarta.</span>
      </div>
      <div class="insight-item">
        <div class="i-dot"></div>
        <span>Model dilatih dengan data scraping listing properti wilayah DKI Jakarta.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close app-wrap