
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import joblib

# Load dataset
df = pd.read_csv("dataset/software_projects.csv")

# Select input features and target
features = [
    "progress_pct",
    "budget_used_pct",
    "team_size",
    "open_bugs",
    "delayed_tasks_pct",
    "requirement_changes",
    "absenteeism_pct"
]

X = df[features]
y = df["risk_level"]

# Convert risk labels to numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=range(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save trained model and label encoder
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/risk_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("\nModel saved successfully!")
