from flask import Flask, request, jsonify
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from flask_cors import CORS 
import numpy as np
from flask_pymongo import PyMongo
from datetime import datetime
import tensorflow as tf
import flask
import os

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb+srv://trueself:trueself@cluster0.ytknj.mongodb.net/trueselfDB?retryWrites=true&w=majority&appName=Cluster0"
mongo = PyMongo(app)

CORS(app)

@app.route('/')
def home():
    try:
        collections = mongo.db.list_collection_names()
        return jsonify({"message": "Successfully runs and  connected to MongoDB!", "collections": collections}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<user_id>', methods=['POST'])
def predict(user_id):
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
        
         # Extract occupation field that is 1
        occupation_fields = ['Occupation_Engineer', 'Occupation_Other', 'Occupation_Student', 
                             'Occupation_Teacher', 'Occupation_Unemployed']
        
        user_occupation = next((field.replace("Occupation_", "") for field in occupation_fields if data.get(field) == 1), None)
        
        mongo.db.anxietyPredictions.insert_one({
            'userId': user_id,
            "severityScore": round(float(predicted_severity[0][0]), 2),
            "age": data['Age'],
            "sleepHours": float(data['Sleep Hours']),
            "physicalActivity": float(data['Physical Activity (hrs/week)']),
            "caffeineIntake": float(data['Caffeine Intake (mg/day)']),
            "alcoholConsumption": float(data['Alcohol Consumption (drinks/week)']),
            "smoking": data['Smoking'],
            "familyHistory": data['Family History of Anxiety'],
            "stressLevel": data['Stress Level (1-10)'],
            "heartRate": int(data['Heart Rate (bpm during attack)']) if isinstance(data['Heart Rate (bpm during attack)'], str) and data['Heart Rate (bpm during attack)'].isdigit() else float(data['Heart Rate (bpm during attack)']),
            "breathingRate": int(data['Breathing Rate (breaths/min)']) if isinstance(data['Breathing Rate (breaths/min)'], str) and data['Breathing Rate (breaths/min)'].isdigit() else float(data['Breathing Rate (breaths/min)']),
            "sweatingLevel": data['Sweating Level (1-5)'],
            "dizziness": data['Dizziness'],
            "medication": data['Medication'],
            "therapySessions": int(data['Therapy Sessions (per month)']) if data['Therapy Sessions (per month)'].isdigit() else float(data['Therapy Sessions (per month)']),
            "recentMajorLifeEvent": data['Recent Major Life Event'],
            "dietQuality": data['Diet Quality (1-10)'], 
            "occupation": user_occupation if user_occupation else "Unknown",
             "createdAt": datetime.utcnow(),
             "updatedAt": datetime.utcnow()
        })
        
        return jsonify({
            'predicted_severity': round(float(predicted_severity[0][0]), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 4000))  
    app.run(debug=False, host="0.0.0.0", port=port)

