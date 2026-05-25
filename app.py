import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="propAI Jakarta",
    page_icon="🏠",
    layout="wide",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  [data-testid="stAppViewContainer"] { background: #050816; }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stSidebar"] { background: #0d1117; }
  .block-container { padding: 2rem 2rem 2rem 2rem; max-width: 1200px; }

  /* Typography */
  h1, h2, h3, h4, p, label, div { color: #e5e7eb !important; }

  /* Cards */
  .card {
    background: #111827;
    border: 1px solid #1e2736;
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }

  /* Result big number */
  .result-amount {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: -0.04em;
    line-height: 1.1;
    text-align: center;
  }
  .result-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    color: #4b5563 !important;
    letter-spacing: .08em;
    text-align: center;
    margin-bottom: 0.5rem;
  }
  .result-sub {
    font-size: 0.75rem;
    color: #4b5563 !important;
    text-align: center;
    margin-top: 0.5rem;
  }

  /* Stat box */
  .stat-box {
    background: #0d1117;
    border: 1px solid #1e2736;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
  }
  .stat-val { font-size: 1.2rem; font-weight: 700; color: #ffffff !important; }
  .stat-lbl { font-size: 0.7rem; color: #4b5563 !important; margin-top: 4px; }

  /* Insight dot */
  .insight-item {
    display: flex;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid #1e2736;
    font-size: 0.82rem;
    color: #6b7280 !important;
    align-items: flex-start;
    line-height: 1.5;
  }
  .insight-item:last-child { border-bottom: none; }
  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #2563eb;
    margin-top: 5px;
    flex-shrink: 0;
  }

  /* Section label */
  .section-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #4b5563 !important;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: 1rem;
  }

  /* Badge */
  .badge {
    display: inline-block;
    background: #111827;
    color: #93c5fd !important;
    border: 1px solid #1e293b;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.72rem;
  }

  /* Streamlit number_input & slider override */
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input {
    background: #0d1117!important;
    border: 1px solid #1e2736!important;
    color: #ffffff!important;
    border-radius: 10px!important;
  }
  [data-testid="stSlider"] [data-baseweb="slider"] { color: #2563eb; }

  /* Button */
  .stButton > button {
    width: 100%;
    background: #2563eb!important;
    color: #fff!important;
    border: none!important;
    border-radius: 12px!important;
    font-size: 1rem!important;
    font-weight: 600!important;
    height: 52px!important;
    transition: background 0.15s;
  }
  .stButton > button:hover { background: #1d4ed8!important; }
</style>
""", unsafe_allow_html=True)

# ─── MODEL CONSTANTS (replika model_rumah.pkl) ─────────────────────────────────
CENTER = np.array([210, 161, 4, 3, 2, 8, 29629629.6])
SCALE  = np.array([200, 154.25, 1, 1, 1, 3, 11973304.5])
COEF   = np.array([-0.01692395, 0.60847173, 0.0253252, 0.01994662,
                    0.02336471, 0.01509061, 0.38533976])
INTERCEPT = 22.12362834422056


def predict_price(lb, lt, kt, km, grs, hlt):
    """Linear Regression with RobustScaler + log-inverse."""
    tr = kt + km
    raw = np.array([lb, lt, kt, km, grs, tr, hlt], dtype=float)
    lp = INTERCEPT + np.sum(COEF * (raw - CENTER) / SCALE)
    return np.expm1(lp)


def fmt_rp(v):
    """Format rupiah into billions (M) or millions (Jt)."""
    if v >= 1e9:
        return f"Rp {v/1e9:.2f} Miliar"
    return f"Rp {v/1e6:,.0f} Jt"


def fmt_short(v):
    if v >= 1e9:
        return f"{v/1e9:.2f} M"
    return f"{v/1e6:,.0f} Jt"


# ─── HEADER ────────────────────────────────────────────────────────────────────
col_logo, col_badge = st.columns([6, 1])
with col_logo:
    st.markdown(
        "<h1 style='font-size:1.8rem;font-weight:500;letter-spacing:-0.03em;"
        "color:#fff!important;margin-bottom:0'>prop<span style='color:#4f8ef7'>AI</span> Jakarta</h1>",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown("<div style='margin-top:0.6rem'><span class='badge'>Machine Learning Model</span></div>",
                unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #1e2736;margin:0.75rem 0 1.5rem'>", unsafe_allow_html=True)

# ─── MAIN LAYOUT ───────────────────────────────────────────────────────────────
left, right = st.columns([1.5, 1], gap="medium")

# ── LEFT: INPUT ────────────────────────────────────────────────────────────────
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Parameter Properti</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        lb = st.number_input("Luas Bangunan (m²)", min_value=0, value=0, step=10, key="lb")
        kt = st.number_input("Kamar Tidur", min_value=0, value=0, step=1, key="kt")
        grs = st.number_input("Garasi", min_value=0, value=0, step=1, key="grs")
    with c2:
        lt = st.number_input("Luas Tanah (m²)", min_value=0, value=0, step=10, key="lt")
        km = st.number_input("Kamar Mandi", min_value=0, value=0, step=1, key="km")

    st.markdown("<br>", unsafe_allow_html=True)

    # Harga/m² tanah — dengan chip buttons
    st.markdown("<label style='font-size:0.85rem;color:#9ca3af'>Harga/m² Tanah (Rp)</label>",
                unsafe_allow_html=True)

    PRESETS = {"15 jt": 15_000_000, "30 jt (median)": 29_629_630,
               "40 jt": 40_000_000, "60 jt": 60_000_000}

    chip_cols = st.columns(4)
    clicked_preset = None
    for i, (label, val) in enumerate(PRESETS.items()):
        with chip_cols[i]:
            if st.button(label, key=f"chip_{i}"):
                clicked_preset = val

    if clicked_preset is not None:
        st.session_state["hlt"] = clicked_preset

    hlt = st.number_input("", min_value=0, value=st.session_state.get("hlt", 0),
                          step=1_000_000, key="hlt", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Prediksi Harga Rumah", key="predict_btn")
    st.markdown("</div>", unsafe_allow_html=True)

# ── RIGHT: RESULT ───────────────────────────────────────────────────────────────
with right:
    # Run prediction
    if predict_clicked or st.session_state.get("has_predicted"):
        st.session_state["has_predicted"] = True
        price = predict_price(lb, lt, kt, km, grs, hlt)
        price_str = fmt_rp(price)
        s1 = fmt_short(price / lb) if lb > 0 else "—"
        s2 = fmt_short(price / lt) if lt > 0 else "—"
        sub_text = "Linear Regression · RobustScaler · Log-inverse"
    else:
        price = None
        price_str = "Rp 0"
        s1, s2 = "—", "—"
        sub_text = "Isi parameter lalu klik prediksi"

    # Result card
    st.markdown(f"""
    <div class='card' style='text-align:center;padding:1.75rem 1.5rem'>
      <div class='result-label'>Estimasi Harga Properti</div>
      <div class='result-amount'>{price_str}</div>
      <div class='result-sub'>{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""
        <div class='stat-box'>
          <div class='stat-val'>{s1}</div>
          <div class='stat-lbl'>Rp/m² bangunan</div>
        </div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class='stat-box'>
          <div class='stat-val'>{s2}</div>
          <div class='stat-lbl'>Rp/m² tanah</div>
        </div>""", unsafe_allow_html=True)

    # Insight card
    st.markdown("""
    <div class='card' style='margin-top:1rem'>
      <div class='section-label'>Insight Model</div>
      <div class='insight-item'>
        <div class='dot'></div>
        <span>LB dan LT menjadi fitur paling berpengaruh terhadap harga rumah.</span>
      </div>
      <div class='insight-item'>
        <div class='dot'></div>
        <span>Model dilatih dengan data scraping listing properti wilayah DKI Jakarta.</span>
      </div>
      <div class='insight-item'>
        <div class='dot'></div>
        <span>Prediksi menggunakan Linear Regression + RobustScaler dengan target log-transform.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── BOTTOM: CHARTS ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
chart_l, chart_r = st.columns(2, gap="medium")

CHART_BG   = "#111827"
CHART_GRID = "rgba(255,255,255,0.07)"
CHART_TICK = "rgba(255,255,255,0.35)"
BLUE_SHADES = ["#2563eb", "#1d4ed8", "#3b82f6", "#60a5fa", "#93c5fd"]

# Bar chart — Distribusi Harga Referensi Jakarta
with chart_l:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Distribusi Harga Referensi Jakarta</div>",
                unsafe_allow_html=True)
    wilayah = ["Jak-Pus", "Jak-Sel", "Jak-Bar", "Jak-Tim", "Jak-Ut"]
    median_harga = [3800, 5200, 2900, 2400, 2100]

    fig_bar = go.Figure(go.Bar(
        x=wilayah, y=median_harga,
        marker_color=BLUE_SHADES,
        text=[f"Rp {v} Jt" for v in median_harga],
        textposition="outside",
        textfont=dict(color=CHART_TICK, size=11),
    ))
    fig_bar.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        margin=dict(l=10, r=10, t=10, b=10), height=220,
        xaxis=dict(tickfont=dict(color=CHART_TICK, size=11), gridcolor=CHART_GRID,
                   showgrid=False, linecolor="#1e2736"),
        yaxis=dict(tickfont=dict(color=CHART_TICK, size=11), gridcolor=CHART_GRID,
                   ticksuffix=" Jt", zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# Line chart — Sensitivitas Harga/m²
with chart_r:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Sensitivitas Harga/m²</div>",
                unsafe_allow_html=True)

    hlt_steps = [10, 15, 20, 25, 30, 40, 50, 60]
    _lb = lb if lb > 0 else 100
    _lt = lt if lt > 0 else 120
    est_prices = [predict_price(_lb, _lt, kt, km, grs, v * 1e6) / 1e6 for v in hlt_steps]

    fig_line = go.Figure(go.Scatter(
        x=[f"{v}jt" for v in hlt_steps], y=est_prices,
        mode="lines+markers",
        line=dict(color="#2563eb", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.15)",
        marker=dict(color="#2563eb", size=5),
    ))
    fig_line.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        margin=dict(l=10, r=10, t=10, b=10), height=220,
        xaxis=dict(tickfont=dict(color=CHART_TICK, size=10), gridcolor=CHART_GRID,
                   showgrid=False, linecolor="#1e2736",
                   title=dict(text="Harga/m² tanah", font=dict(color=CHART_TICK, size=11))),
        yaxis=dict(tickfont=dict(color=CHART_TICK, size=11), gridcolor=CHART_GRID,
                   ticksuffix=" Jt", zeroline=False),
        showlegend=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;font-size:0.72rem;color:#374151'>
  propAI Jakarta · Linear Regression + RobustScaler · Data: DKI Jakarta Property Listings
</div>
""", unsafe_allow_html=True)