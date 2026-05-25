import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from pathlib import Path

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="propAI Jakarta", page_icon="🏠", layout="wide")

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"] {
  background: #1a1c23 !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }

.result-card {
  background: #22253a; border: 1px solid #2c2f42;
  border-radius: 14px; padding: 2rem 1.5rem 1.5rem; text-align: center;
  margin-bottom: 1.25rem;
}
.result-title {
  font-size: .65rem; font-weight: 700; color: #5a6070 !important;
  text-transform: uppercase; letter-spacing: .12em; margin-bottom: .6rem;
}
.result-value {
  font-size: 2.4rem; font-weight: 700; color: #e8eaf0 !important;
  letter-spacing: -.04em; line-height: 1;
}
.result-sub { font-size: .7rem; color: #5a6070 !important; margin-top: .5rem; }

.stats-row {
  display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; margin-bottom: 1.25rem;
}
.stat-box {
  background: #161820; border: 1px solid #2c2f42;
  border-radius: 10px; padding: 1rem; text-align: center;
}
.stat-val { font-size: 1rem; font-weight: 700; color: #e8eaf0 !important; }
.stat-lbl { font-size: .65rem; color: #5a6070 !important; margin-top: 3px; }

.insight-card {
  background: #22253a; border: 1px solid #2c2f42;
  border-radius: 14px; padding: 1.3rem 1.4rem;
}
.sec-label {
  font-size: .65rem; font-weight: 700; color: #5a6070 !important;
  text-transform: uppercase; letter-spacing: .12em; margin-bottom: .75rem;
}
.insight-item {
  display: flex; gap: 10px; padding: 8px 0;
  border-bottom: 1px solid #2c2f42; font-size: .78rem;
  color: #8b90a0 !important; align-items: flex-start; line-height: 1.5;
}
.insight-item:last-child { border-bottom: none; padding-bottom: 0; }
.i-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: #4f8ef7; margin-top: 5px; flex-shrink: 0;
}

/* inputs */
div[data-testid="stNumberInput"] label {
  font-size: .78rem !important; color: #8b90a0 !important;
  font-family: 'DM Sans', sans-serif !important; margin-bottom: 4px !important;
}
div[data-testid="stNumberInput"] > div {
  background: #161820 !important; border: 1px solid #2c2f42 !important;
  border-radius: 8px !important; overflow: hidden !important;
}
div[data-testid="stNumberInput"] input {
  background: transparent !important; color: #e8eaf0 !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .95rem !important; font-weight: 500 !important;
  border: none !important; padding: 10px 12px !important;
}
div[data-testid="stNumberInput"] button {
  background: transparent !important; border: none !important; color: #5a6070 !important;
}
div[data-testid="stNumberInput"] button:hover { color: #e8eaf0 !important; }

/* chip-style small buttons */
div[data-testid="stHorizontalBlock"] button {
  background: #161820 !important; border: 1px solid #2c2f42 !important;
  color: #8b90a0 !important; border-radius: 999px !important;
  font-size: .72rem !important; font-family: 'DM Sans', sans-serif !important;
  padding: 4px 10px !important; height: auto !important; min-height: 0 !important;
  font-weight: 500 !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
  background: #1c2b55 !important; color: #93c5fd !important;
  border-color: #2563eb !important;
}

/* predict button */
div[data-testid="stButton"]:not(div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]) button {
  width: 100% !important; background: #2c2f42 !important; color: #e8eaf0 !important;
  border: 1px solid #3a3f58 !important; border-radius: 10px !important;
  font-size: .95rem !important; font-weight: 600 !important; height: 48px !important;
  font-family: 'DM Sans', sans-serif !important; letter-spacing: -.01em !important;
}
div[data-testid="stButton"]:not(div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]) button:hover {
  background: #363b56 !important;
}

/* section heading */
p[style*="sec-label"], .sec-h {
  font-size: .65rem !important; font-weight: 700 !important; color: #5a6070 !important;
  text-transform: uppercase; letter-spacing: .12em;
}

#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL & SCALER ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load model_rumah.pkl and scaler.pkl.
    Letakkan kedua file di folder yang sama dengan propai_jakarta.py."""
    base = Path(__file__).parent
    with open(base / "model_rumah.pkl", "rb") as f:
        model = pickle.load(f)
    with open(base / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
    MODEL_OK = True
except FileNotFoundError as e:
    MODEL_OK = False
    LOAD_ERR = str(e)

# ── PREDICTION HELPER ───────────────────────────────────────────────────────────
def predict(lb, lt, kt, km, grs, harga_per_lt):
    """Predict house price using real pkl artifacts.
    
    Features (must match training order):
      LB, LT, KT, KM, GRS, TOTAL_RUANG, HARGA_PER_LT
    """
    total_ruang = kt + km
    X = pd.DataFrame([[lb, lt, kt, km, grs, total_ruang, harga_per_lt]],
                     columns=["LB","LT","KT","KM","GRS","TOTAL_RUANG","HARGA_PER_LT"])
    X_scaled = scaler.transform(X)
    pred_log  = model.predict(X_scaled)
    return float(np.expm1(pred_log)[0])

def fmt_rp(v):
    if v >= 1e9: return f"Rp {v/1e9:.2f} M"
    return f"Rp {v/1e6:,.0f} Jt"

def fmt_short(v):
    if v >= 1e9: return f"{v/1e9:.2f} M"
    return f"{v/1e6:,.0f} Jt"

# ── CHART THEME ──────────────────────────────────────────────────────────────────
GRID = "rgba(255,255,255,0.05)"
TICK = "rgba(200,205,220,0.45)"

# ── TOPBAR ───────────────────────────────────────────────────────────────────────
c_logo, c_badge = st.columns([6, 1])
with c_logo:
    st.markdown(
        "<h1 style='font-size:1.35rem;font-weight:600;color:#e8eaf0;"
        "letter-spacing:-.02em;font-family:DM Sans,sans-serif;margin:1.5rem 0 .2rem 1.5rem'>"
        "prop<span style=\"color:#4f8ef7\">AI</span> Jakarta</h1>",
        unsafe_allow_html=True)
with c_badge:
    st.markdown(
        "<div style='margin-top:1.7rem;margin-right:1rem;text-align:right'>"
        "<span style='background:#22253a;color:#8ba7e8;border:1px solid #2c3555;"
        "border-radius:999px;padding:5px 16px;font-size:.72rem;font-weight:500'>"
        "Machine Learning Model</span></div>",
        unsafe_allow_html=True)

st.markdown(
    "<hr style='border:none;border-top:1px solid #2c2f3a;margin:.3rem 1.5rem 1.5rem'>",
    unsafe_allow_html=True)

if not MODEL_OK:
    st.error(f"❌ Gagal memuat model: {LOAD_ERR}\n\n"
             "Pastikan `model_rumah.pkl` dan `scaler.pkl` berada di folder yang sama dengan `propai_jakarta.py`.")
    st.stop()

# ── MAIN 2-COLUMN LAYOUT ─────────────────────────────────────────────────────────
pad = "padding: 0 1.5rem"
col_l, col_r = st.columns([1, 1], gap="large")

# ═══════════════ LEFT ════════════════════════════════════════════════════════════
with col_l:
    st.markdown(f"<div style='{pad}'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:.65rem;font-weight:700;color:#5a6070;"
        "text-transform:uppercase;letter-spacing:.12em;margin-bottom:1rem'>"
        "Parameter Properti</p>", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2, gap="medium")
    with r1c1:
        lb  = st.number_input("Luas bangunan (m²)", min_value=0, value=100, step=10)
        kt  = st.number_input("Kamar tidur",         min_value=0, value=3,   step=1)
        grs = st.number_input("Garasi",               min_value=0, value=1,   step=1)
    with r1c2:
        lt  = st.number_input("Luas tanah (m²)",     min_value=0, value=120, step=10)
        km  = st.number_input("Kamar mandi",          min_value=0, value=2,   step=1)

    # Harga per LT
    st.markdown(
        "<p style='font-size:.78rem;color:#8b90a0;margin:.8rem 0 .4rem'>"
        "Harga/m² tanah — HARGA_PER_LT (Rp)</p>",
        unsafe_allow_html=True)

    PRESETS = {"15 jt": 15_000_000, "30 jt (median)": 29_629_630,
               "40 jt": 40_000_000, "60 jt": 60_000_000}
    chip_cols = st.columns(4, gap="small")
    for i, (lbl, val) in enumerate(PRESETS.items()):
        with chip_cols[i]:
            if st.button(lbl, key=f"chip_{i}"):
                st.session_state["hlt"] = val

    hlt = st.number_input(
        "", min_value=0,
        value=st.session_state.get("hlt", 5_000_000),
        step=1_000_000, label_visibility="collapsed")
    st.session_state["hlt"] = hlt   # keep in sync

    st.markdown("<div style='margin-top:.75rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Prediksi harga", key="pred_btn")

    # ── Bar chart distribusi wilayah ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:.65rem;font-weight:700;color:#5a6070;"
        "text-transform:uppercase;letter-spacing:.12em;margin-bottom:.6rem'>"
        "Distribusi Harga Referensi Jakarta</p>",
        unsafe_allow_html=True)

    wilayah  = ["Jak-Pus","Jak-Sel","Jak-Bar","Jak-Tim","Jak-Ut"]
    med_h    = [3800, 5200, 2900, 2400, 2100]
    bar_colors = ["#4f8ef7","#3b7de8","#6aa3f9","#8dbeff","#b0d4ff"]

    fig_bar = go.Figure(go.Bar(
        x=med_h, y=wilayah, orientation="h",
        marker_color=bar_colors,
        text=[f"{v/1000:.1f}M" if v >= 1000 else f"{v}Jt" for v in med_h],
        textposition="outside",
        textfont=dict(color=TICK, size=10),
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=30, t=0, b=0), height=155,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(color=TICK, size=11), showgrid=False,
                   categoryorder="array", categoryarray=list(reversed(wilayah))),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════ RIGHT ═══════════════════════════════════════════════════════════
with col_r:
    st.markdown(f"<div style='{pad}'>", unsafe_allow_html=True)

    # ── Run prediction ────────────────────────────────────────────────────────
    if predict_btn:
        price = predict(lb, lt, kt, km, grs, hlt)
        st.session_state.update({
            "has_pred": True, "price": price,
            "lb": lb, "lt": lt
        })

    if st.session_state.get("has_pred"):
        price    = st.session_state["price"]
        _lb, _lt = st.session_state.get("lb", lb), st.session_state.get("lt", lt)
        p_str    = fmt_rp(price)
        s1 = fmt_short(price / _lb) if _lb > 0 else "—"
        s2 = fmt_short(price / _lt) if _lt > 0 else "—"
        sub  = "Model: Regression · Scaled · Log-inverse"
        body = sub
    else:
        price = None; p_str = "—"; s1 = s2 = "—"
        body = "Isi parameter &amp; klik prediksi"
        sub  = "Model: Regression · Scaled · Log-inverse"

    st.markdown(f"""
    <div class="result-card">
      <div class="result-title">Estimasi Harga Properti</div>
      <div class="result-value">{p_str}</div>
      <div class="result-sub">{body}</div>
      <div class="result-sub">{sub if st.session_state.get("has_pred") else ""}</div>
    </div>""", unsafe_allow_html=True)

    # ── Tren Harga Simulasi ───────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:.65rem;font-weight:700;color:#5a6070;"
        "text-transform:uppercase;letter-spacing:.12em;margin-bottom:.5rem'>"
        "Tren Harga Simulasi</p>", unsafe_allow_html=True)

    months = ["Jan","Feb","Mar","Apr","Mei","Jun"]
    _lb2  = st.session_state.get("lb", lb) or 100
    _lt2  = st.session_state.get("lt", lt) or 120
    base_hlt = hlt if hlt > 0 else 5_000_000

    trend = [predict(_lb2, _lt2, kt, km, grs, base_hlt * (1 + 0.04*i)) / 1e6
             for i in range(6)]

    fig_line = go.Figure(go.Scatter(
        x=months, y=trend, mode="lines+markers",
        line=dict(color="#4f8ef7", width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.12)",
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

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-box">
        <div class="stat-val">{s1}</div>
        <div class="stat-lbl">Rp/m² bangunan</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">{s2}</div>
        <div class="stat-lbl">Rp/m² tanah</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Insight ───────────────────────────────────────────────────────────────
    coef_names = ["LB","LT","KT","KM","GRS","TOTAL_RUANG","HARGA_PER_LT"]
    coefs      = model.coef_
    top2       = sorted(zip(coef_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:2]
    top2_str   = " &amp; ".join(f"<b>{n}</b>" for n, _ in top2)

    st.markdown(f"""
    <div class="insight-card">
      <div class="sec-label">Insight Model</div>
      <div class="insight-item">
        <div class="i-dot"></div>
        <span>{top2_str} adalah fitur terkuat dalam model prediksi harga rumah Jakarta.</span>
      </div>
      <div class="insight-item">
        <div class="i-dot"></div>
        <span>Harga per m² tanah bervariasi signifikan antar kecamatan di Jakarta.</span>
      </div>
      <div class="insight-item">
        <div class="i-dot"></div>
        <span>Model dilatih dengan data scraping listing properti wilayah DKI Jakarta.</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)