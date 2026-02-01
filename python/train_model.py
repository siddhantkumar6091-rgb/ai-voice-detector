import os
import numpy as np
from feature_extractor import extract_features
from sklearn.ensemble import RandomForestClassifier
import joblib

X = []
y = []

# HUMAN = 0, AI = 1
for label, folder in enumerate(["human", "ai"]):
    folder_path = os.path.join("audio", folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        features = extract_features(file_path)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "model/voice_model.pkl")

print("✅ Model trained and saved!")
