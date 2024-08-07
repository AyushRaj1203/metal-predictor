from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from CBFV import composition
from sklearn.preprocessing import StandardScaler, normalize
import os


app = Flask(__name__, static_folder='static', template_folder='templates')

# Path to the model file
MODEL_PATH = 'model.pkl'
 
# Load the model and scaler
model_loaded = joblib.load(MODEL_PATH)
model = model_loaded['model']
scaler = model_loaded['scaler']


# Function to predict metal or non-metal
def predict_metal(formula):
    input_data = pd.DataFrame({'formula': [formula], 'target': [0]})
    try:
        features, _, _, _ = composition.generate_features(input_data, elem_prop='magpie', drop_duplicates=False, extend_features=True, sum_feat=True)
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
    prediction, error = predict_metal(formula)
    return jsonify({'prediction': prediction, 'error': error})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
