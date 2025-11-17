import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import os

# Load Data
df = pd.read_csv("data/credit_data.csv")

# Drop unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# Features and target
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate Model
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/credit_model.pkl")
print("Model Saved!")
