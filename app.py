from flask import Flask
from flask import render_template
from flask import request

import numpy as np
import pandas as pd

import pickle

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(
    open('model_rumah.pkl', 'rb')
)

scaler = pickle.load(
    open('scaler.pkl', 'rb')
)

# =========================
# HOME
# =========================
@app.route('/')
def home():
    return render_template('index.html')

# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])

def predict():

    LB = float(request.form['LB'])
    LT = float(request.form['LT'])
    KT = float(request.form['KT'])
    KM = float(request.form['KM'])
    GRS = float(request.form['GRS'])

    TOTAL_RUANG = KT + KM

    HARGA_PER_LT = 5000000

    data = pd.DataFrame({

        'LB': [LB],
        'LT': [LT],
        'KT': [KT],
        'KM': [KM],
        'GRS': [GRS],
        'TOTAL_RUANG': [TOTAL_RUANG],
        'HARGA_PER_LT': [HARGA_PER_LT]

    })

    # scaling
    data_scaled = scaler.transform(data)

    # predict
    pred_log = model.predict(data_scaled)

    # inverse log
    harga = np.expm1(pred_log)[0]

    # format
    if harga >= 1_000_000_000:

        hasil = f"{harga/1_000_000_000:.2f} M"

    else:

        hasil = f"{harga/1_000_000:.2f} JT"

    return render_template(
        'index.html',
        prediction=hasil
    )

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=False)