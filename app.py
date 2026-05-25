# =========================================================
# IMPORT LIBRARY
# =========================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pickle

# =========================================================
# FEATURE ENGINEERING
# =========================================================

# TOTAL RUANG
df_handled['TOTAL_RUANG'] = (
    df_handled['KT'] +
    df_handled['KM']
)

# =========================================================
# FITUR & TARGET
# =========================================================

# HAPUS HARGA_PER_LT
# karena menyebabkan data leakage

X = df_handled[[

    'LB',
    'LT',
    'KT',
    'KM',
    'GRS',
    'TOTAL_RUANG'

]]

y = df_handled['HARGA']

# =========================================================
# TRANSFORMASI LOG TARGET
# =========================================================
y_log = np.log1p(y)

# =========================================================
# SPLIT DATA
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_log,

    test_size=0.2,
    random_state=42

)

# =========================================================
# SCALING
# =========================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# =========================================================
# MODEL
# =========================================================
model = LinearRegression()

model.fit(
    X_train_scaled,
    y_train
)

# =========================================================
# PREDIKSI
# =========================================================
y_pred_log = model.predict(
    X_test_scaled
)

# balik ke harga asli
y_pred = np.expm1(
    y_pred_log
)

y_asli = np.expm1(
    y_test
)

# =========================================================
# EVALUASI
# =========================================================
mae = mean_absolute_error(
    y_asli,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_asli,
        y_pred
    )
)

r2 = r2_score(
    y_asli,
    y_pred
)

# =========================================================
# HASIL
# =========================================================
hasil = pd.DataFrame({

    'MAE': [f"Rp {mae:,.0f}"],
    'RMSE': [f"Rp {rmse:,.0f}"],
    'R2 Score': [round(r2, 4)]

})

print(hasil)

# =========================================================
# SAVE MODEL
# =========================================================
pickle.dump(
    model,
    open('model_rumah.pkl', 'wb')
)

pickle.dump(
    scaler,
    open('scaler.pkl', 'wb')
)

print("Model berhasil disimpan")