import pickle 
from flask import Flask, request, jsonify, render_template

import numpy as np
import pandas as pd


app = Flask(__name__)

# Load the model and scaler safely with error handling
try:
    xgbmodel = pickle.load(open('lgb_model.pkl', 'rb'))
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    xgbmodel = None

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
    if xgbmodel is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded properly.'}), 500

    data = request.json['data']
    print(data)
    input_array = np.array(list(data.values())).reshape(1, -1)
    print(input_array)
    new_data = scaler.transform(input_array)
    output = xgbmodel.predict(new_data)
    print(output[0])
    return jsonify({'prediction': output[0]})

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True, port=5050)
