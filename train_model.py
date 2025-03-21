import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Load Dataset
data = pd.read_csv("dataset/heart.csv")
print("Dataset Preview:\n", data.head())
print("\nInitial Class Distribution:\n", data["target"].value_counts())

# ✅ Fix: Ensure both classes 0 and 1 have at least 2 samples
class_counts = data["target"].value_counts()
if class_counts.min() < 2:
    print("⚠️ Warning: One class has too few samples! Duplicating data to fix this issue.")

    # Find the class with the least samples
    min_class = class_counts.idxmin()

    # Duplicate rows of the minority class until it has at least 2 samples
    while data["target"].value_counts()[min_class] < 2:
        data = pd.concat([data, data[data["target"] == min_class]], ignore_index=True)

    # Save the updated dataset
    data.to_csv("dataset/heart.csv", index=False)
    print("\n✅ Fixed: Updated Class Distribution:\n", data["target"].value_counts())

# Splitting Features and Target
X = data.drop(columns=["target"])
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# ✅ Create "model" directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# ✅ Save Model
joblib.dump(model, "model/heart_disease_model.pkl")
print("✅ Model saved successfully: model/heart_disease_model.pkl")
