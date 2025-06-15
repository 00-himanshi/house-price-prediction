import pickle 
from flask import Flask, request, jsonify, render_template, url_for

import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ Define the actual feature columns (excluding MEDV, the target)
FEATURE_COLUMNS = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
    'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT'
]

# Load the model and scaler safely with error handling
try:
    lgbmodel = pickle.load(open('lgb_model.pkl', 'rb'))
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    lgbmodel = None

try:
    scaler = pickle.load(open('scaling.pkl', 'rb'))
    print("✅ Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if lgbmodel is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded properly.'}), 500

    data = request.json['data']
    print(data)

    # Convert input into a DataFrame with correct column names
    input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    print(input_df)

    # Transform and predict
    new_data = scaler.transform(input_df)
    output = lgbmodel.predict(new_data)
    print(output[0])

    return jsonify({'prediction': output[0]})

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]

    # Convert to DataFrame with column names
    input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    print(input_df)

    final_input = scaler.transform(input_df)
    output = lgbmodel.predict(final_input)[0]

    return render_template('index.html', prediction_text="The House Price prediction is: {}".format(output))

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True, port=5050)
