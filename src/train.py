import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.features import FEATURES

df = pd.read_csv("data/creditcard.csv")
df = df[FEATURES].dropna()

scaler = StandardScaler()
X = scaler.fit_transform(df)

model = KMeans(n_clusters=4, random_state=42, n_init=10)
model.fit(X)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Training complete (production-ready)")