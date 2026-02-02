import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("balanced_data.csv")

# Cleaning
df.drop(columns=["patient_id"], inplace=True)

df[["systolic_bp", "diastolic_bp"]] = (
    df["blood_pressure"].str.split("/", expand=True).astype(int)
)
df.drop(columns=["blood_pressure"], inplace=True)

binary_cols = ["diabetes", "hypertension", "readmitted_30_days"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

X = df.drop("readmitted_30_days", axis=1)
y = df["readmitted_30_days"]

num_cols = [
    "age", "cholesterol", "bmi",
    "medication_count", "length_of_stay",
    "systolic_bp", "diastolic_bp"
]
cat_cols = ["gender", "discharge_destination"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)

# SAVE WHOLE PIPELINE
joblib.dump(model, "readmission_model.pkl")

print("✅ Model + Encoder + Scaler saved as readmission_model.pkl")