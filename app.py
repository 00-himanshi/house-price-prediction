import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ Define the feature columns in correct order
FEATURE_COLUMNS = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
    'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT'
]

# ✅ Load model and scaler
try:
    with open('lgb_model.pkl', 'rb') as file:
        lgbmodel = pickle.load(file)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    lgbmodel = None

try:
    with open('scaling.pkl', 'rb') as file:
        scaler = pickle.load(file)
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

    try:
        data = request.json['data']
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        # ✅ Scale input and predict
        scaled_data = scaler.transform(input_df)
        prediction = lgbmodel.predict(scaled_data)[0]

        return jsonify({'prediction': round(float(prediction), 2)})
    except Exception as e:
        print("❌ API prediction error:", e)
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    if lgbmodel is None or scaler is None:
        return render_template('index.html', prediction_text="Model or scaler not loaded properly.")

    try:
        # ✅ Safely collect and order input
        form_data = [float(request.form.get(col, 0)) for col in FEATURE_COLUMNS]
        input_df = pd.DataFrame([form_data], columns=FEATURE_COLUMNS)

        scaled_input = scaler.transform(input_df)
        prediction = lgbmodel.predict(scaled_input)[0]

        return render_template('index.html', prediction_text=f"The House Price prediction is: {prediction:.2f}")
    except Exception as e:
        print("❌ Error during form prediction:", e)
        return render_template('index.html', prediction_text="Error processing input. Check values and try again.")

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True, port=5050)
