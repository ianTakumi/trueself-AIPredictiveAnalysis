from flask import Flask, request, jsonify
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()
        # Load the model and scaler
        model = load_model("anxiety_model.h5") 
        scaler = joblib.load("scaler.pkl")  
        
        required_fields = [
            'Age', 'Sleep Hours', 'Physical Activity (hrs/week)',
            'Caffeine Intake (mg/day)', 'Alcohol Consumption (drinks/week)',
            'Smoking', 'Family History of Anxiety', 'Stress Level (1-10)',
            'Heart Rate (bpm during attack)', 'Breathing Rate (breaths/min)',
            'Sweating Level (1-5)', 'Dizziness', 'Medication',
            'Therapy Sessions (per month)', 'Recent Major Life Event',
            'Diet Quality (1-10)', 'Occupation_Engineer', 'Occupation_Other',
            'Occupation_Student', 'Occupation_Teacher', 'Occupation_Unemployed'
        ]
        
        # Check for missing field
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400 
        
        # Convert input data to Dataframe
        df_test = pd.DataFrame([data])
        
        # Normalize the data
        X_test_scaled = scaler.transform(df_test)
        
        # Make a prediction
        predicted_severity = model.predict(X_test_scaled)
        
        return jsonify({
            'predicted_severity': round(float(predicted_severity[0][0]), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=4000)
