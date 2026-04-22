import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Clean data
df = df.drop(columns=["CUST_ID"], errors="ignore")
df = df.fillna(df.mean(numeric_only=True))

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Save model + scaler
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model training complete ✔")