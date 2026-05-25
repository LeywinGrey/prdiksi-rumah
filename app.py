import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open('model_rumah.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Median HARGA_PER_LT dari data training (RobustScaler center)
DEFAULT_HARGA_PER_LT = 29_629_629

# =========================
# TITLE
# =========================
st.title("Prediksi Harga Rumah Wilayah Jakarta")
st.write("Masukkan data rumah untuk memprediksi harga.")

# =========================
# INPUT USER
# =========================
LB = st.number_input("Luas Bangunan (m²)", min_value=1)
LT = st.number_input("Luas Tanah (m²)", min_value=1)
KT = st.number_input("Kamar Tidur", min_value=1)
KM = st.number_input("Kamar Mandi", min_value=1)
GRS = st.number_input("Garasi", min_value=0)

# =========================
# HARGA PER METER TANAH (OPSIONAL)
# =========================
with st.expander("➕ Masukkan Harga per Meter Tanah (opsional)"):
    st.caption(
        "Jika tidak diisi, harga per meter tanah akan menggunakan "
        "nilai median dari data training yaitu "
        "Rp 29.629.629 per meter persegi."
    )
    gunakan_harga_manual = st.checkbox("Saya ingin memasukkan harga sendiri")

    if gunakan_harga_manual:
        HARGA_PER_LT = st.number_input(
            "Harga per Meter Persegi Tanah (Rp/m²)",
            min_value=0,
            value=0,
            step=500_000,
            format="%d",
            help="Contoh: jika harga tanah Rp 5.000.000 per m², masukkan 5000000"
        )

        # Format Rp yang mudah dibaca
        def format_rupiah(angka):
            if angka >= 1_000_000_000:
                return f"Rp {angka/1_000_000_000:.2f} miliar per meter persegi"
            elif angka >= 1_000_000:
                miliar_part = int(angka) // 1_000_000
                juta_sisa = int(angka) % 1_000_000 // 1_000
                if juta_sisa > 0:
                    return f"Rp {miliar_part}.{juta_sisa:03d}.000 per meter persegi"
                else:
                    return f"Rp {miliar_part}.000.000 per meter persegi"
            else:
                return f"Rp {int(angka):,} per meter persegi".replace(",", ".")

        if HARGA_PER_LT > 0:
            st.info(f"📐 Harga per Meter Tanah: **{format_rupiah(HARGA_PER_LT)}**")
    else:
        HARGA_PER_LT = DEFAULT_HARGA_PER_LT

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

    rumah_scaled = scaler.transform(rumah_baru)
    pred_log = model.predict(rumah_scaled)
    harga = np.expm1(pred_log)[0]

    if harga >= 1_000_000_000:
        hasil = f"Rp {harga/1_000_000_000:.2f} miliar"
    else:
        hasil = f"Rp {harga/1_000_000:.2f} juta"

    st.success(f"Prediksi Harga Rumah: {hasil}")