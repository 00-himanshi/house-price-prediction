import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ Define the feature columns used during training
FEATURE_COLUMNS = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
    'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO',        'B', 
    'LSTAT'
]

# ✅ Load model and scaler
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

    # ✅ Ensure it has all 13 features
    input_df = pd.DataFrame([data])
    input_df = input_df[FEATURE_COLUMNS]  # Reorder and filter
    print(input_df)

    new_data = scaler.transform(input_df)
    output = lgbmodel.predict(new_data)

    return jsonify({'prediction': output[0]})

@app.route('/predict', methods=['POST'])
def predict():
    if lgbmodel is None or scaler is None:
        return render_template('index.html', prediction_text="Model or scaler not loaded properly.")

    try:
        form_values = [float(x) for x in request.form.values()]
        input_df = pd.DataFrame([form_values], columns=FEATURE_COLUMNS)
        print(input_df)

        final_input = scaler.transform(input_df)
        output = lgbmodel.predict(final_input)[0]

        return render_template('index.html', prediction_text=f"The House Price prediction is: {output:.2f}")
    except Exception as e:
        print("❌ Error during prediction:", e)
        return render_template('index.html', prediction_text="Error processing the input.")

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True, port=5050)
