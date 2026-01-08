# minimal_model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import os

# --- CONFIG ---
DATA_PATH = r"E:\Project\alzheimers_disease_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- LOAD & CLEAN DATA ---
df = pd.read_csv(DATA_PATH)

# Drop non-predictive columns
df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')

# Convert object columns to numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with median
df = df.fillna(df.median())

# --- FEATURES & TARGET ---
target_col = 'Diagnosis'
X = df.drop(columns=[target_col])
y = df[target_col]

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- PIPELINE WITH IMPUTER, SMOTE, SCALER, RANDOM FOREST ---
pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        class_weight='balanced'
    ))
])

# --- TRAIN MODEL ---
pipeline.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- SAVE MODEL ---
with open(os.path.join(MODEL_DIR, "rf_alzheimers_model.pkl"), "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved successfully.")
