import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Segmentation App",
    layout="wide"
)

# =========================
# STYLE (FINTECH LOOK)
# =========================
st.markdown("""
<style>
.main { background-color: #0e1117; }
h1 { color: #00ffd5; text-align: center; }
</style>
""", unsafe_allow_html=True)

# =========================
# FEATURES
# =========================
FEATURES = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY"
]

MODEL_PATH = "models/kmeans_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# =========================
# LOAD OR TRAIN MODEL (AUTO FIX)
# =========================
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

model, scaler = load_model()

# =========================
# AUTO TRAIN IF MISSING
# =========================
def train_model(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    model.fit(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler

# =========================
# HEADER
# =========================
st.title("💳 Customer Segmentation System")
st.markdown("AI-powered customer clustering using KMeans + PCA")

# =========================
# UPLOAD DATA
# =========================
file = st.file_uploader("Upload Customer CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    # =========================
    # VALIDATION
    # =========================
    missing = [c for c in FEATURES if c not in df.columns]

    if missing:
        st.error(f"""
❌ Missing required columns:
{missing}

Required columns:
{FEATURES}
""")
        st.stop()

    data = df[FEATURES].dropna()

    # =========================
    # AUTO TRAIN IF NEEDED
    # =========================
    if model is None or scaler is None:
        st.warning("⚠ Model not found. Training new model...")
        model, scaler = train_model(data)

    # =========================
    # PREDICTION
    # =========================
    scaled = scaler.transform(data)
    clusters = model.predict(scaled)

    df = df.loc[data.index].copy()
    df["Cluster"] = clusters

    # =========================
    # LABELS
    # =========================
    cluster_map = {
        0: "Low Activity Users",
        1: "Cash Advance Users",
        2: "High Spenders",
        3: "VIP Customers"
    }

    df["Segment"] = df["Cluster"].map(cluster_map)

    # =========================
    # KPI DASHBOARD
    # =========================
    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", len(df))
    col2.metric("VIP", sum(df["Segment"] == "VIP Customers"))
    col3.metric("High Spenders", sum(df["Segment"] == "High Spenders"))
    col4.metric("Others", sum(df["Cluster"] <= 1))

    # =========================
    # DATA PREVIEW
    # =========================
    st.subheader("📄 Segmented Data")
    st.dataframe(df.head())

    # =========================
    # PCA VISUALIZATION
    # =========================
    st.subheader("📍 Customer Clusters (PCA View)")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=clusters,
        cmap="viridis",
        alpha=0.7
    )

    ax.set_title("Customer Segmentation Map")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    st.pyplot(fig)

    # =========================
    # BUSINESS INSIGHTS
    # =========================
    st.subheader("📊 Business Insights")

    for seg, count in df["Segment"].value_counts().items():
        st.write(f"**{seg}** → {count} customers")

    st.success("✅ Analysis completed successfully")