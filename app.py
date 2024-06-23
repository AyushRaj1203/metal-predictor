from flask import Flask, render_template, request, jsonify
import json
import pickle
import pandas as pd
from CBFV import composition
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import re
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the model and scaler from the pickle file
with open('model.pkl', 'rb') as file:
    loaded_model_info = pickle.load(file)

model_loaded = loaded_model_info["model"]
scaler_loaded = loaded_model_info["scaler"]

# Function to predict metal or non-metal
def predict_metal(formula, model, scaler):
    input = pd.DataFrame({'formula': [formula], 'target': [0]})
    try:
        features, _, _, _ = composition.generate_features(input, elem_prop='magpie', drop_duplicates=False, extend_features=True, sum_feat=True)
    except ValueError as e:
        if 'not in list' in str(e):
            return None, "Enter a valid Formula!!"
        else:
            raise e
    features_scaled = scaler.transform(features)
    features_normalized = normalize(features_scaled)
    prediction = model.predict(features_normalized)
    prediction_label = 'Metal' if prediction[0] == 1 else 'Non-Metal'
    return prediction_label, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    formula = request.form['formula']
    prediction, error = predict_metal(formula, model_loaded, scaler_loaded)
    return jsonify({'prediction': prediction, 'error': error})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
