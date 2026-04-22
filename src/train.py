import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# =========================
# 1. Load Data
# =========================
df = pd.read_csv("data/creditcard.csv")

# =========================
# 2. Clean Data
# =========================
df = df.drop(columns=["CUST_ID"], errors="ignore")  # remove ID
df = df.fillna(df.mean(numeric_only=True))          # fill missing values

# =========================
# 3. Scale Features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================
# 4. Train Model
# =========================
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# =========================
# 5. Save Model
# =========================
os.makedirs("models", exist_ok=True)

joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved successfully!")