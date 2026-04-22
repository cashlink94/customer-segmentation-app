import joblib
import pandas as pd
from src.features import FEATURES

model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

cluster_map = {
    0: "Low Activity Users",
    1: "Cash Advance Users",
    2: "High Spenders",
    3: "VIP Customers"
}

def predict_segments(df: pd.DataFrame):
    data = df[FEATURES].dropna()
    scaled = scaler.transform(data)

    clusters = model.predict(scaled)

    df = df.copy()
    df["Cluster"] = clusters
    df["Segment"] = df["Cluster"].map(cluster_map)

    return df