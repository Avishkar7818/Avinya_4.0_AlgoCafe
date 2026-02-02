import joblib
import pandas as pd
model = joblib.load("readmission_model.pkl")

sample = {
    "age": 28,
    "gender": "Female",
    "cholesterol": 160,
    "bmi": 22.4,
    "diabetes": 0,
    "hypertension": 0,
    "medication_count": 1,
    "length_of_stay": 1,
    "discharge_destination": "Home",
    "systolic_bp": 112,
    "diastolic_bp": 72
}


df = pd.DataFrame([sample])

risk = model.predict_proba(df)[0][1]
print("Readmission Risk:", round(risk, 2))