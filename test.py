import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model and scaler
model = load_model("anxiety_model.h5") 
scaler = joblib.load("scaler.pkl")  

# Create a DataFrame from the test data
test_data = {
    'Age': [23],
    'Sleep Hours': [9.8],
    'Physical Activity (hrs/week)': [8.1],
    'Caffeine Intake (mg/day)': [140],
    'Alcohol Consumption (drinks/week)': [19],
    'Smoking': [1],
    'Family History of Anxiety': [0],  
    'Stress Level (1-10)': [2],
    'Heart Rate (bpm during attack)': [81],
    'Breathing Rate (breaths/min)': [33],
    'Sweating Level (1-5)': [2],
    'Dizziness': [0],
    'Medication': [0], 
    'Therapy Sessions (per month)': [8],
    'Recent Major Life Event': [0], 
    'Diet Quality (1-10)': [1],
    'Occupation_Engineer': [0], 
    'Occupation_Other': [0],  
    'Occupation_Student': [1],  
    'Occupation_Teacher': [0], 
    'Occupation_Unemployed': [0],
}

# Convert the test data to a DataFrame
df_test = pd.DataFrame(test_data)

# Normalize the test data using the same scaler used for training
X_test_scaled = scaler.transform(df_test)

# Make a prediction with the trained model
predicted_severity = model.predict(X_test_scaled)
print(predicted_severity)


print(f"Predicted Severity of Anxiety Attack: {predicted_severity[0][0]:.2f}")
