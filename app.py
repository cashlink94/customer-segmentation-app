import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("💳 Customer Segmentation App")
st.markdown("AI-powered customer segmentation using KMeans clustering.")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    kmeans = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return kmeans, scaler

try:
    model, scaler = load_model()
except:
    st.error("❌ Model not found. Please run training first (src/train.py).")
    st.stop()

# =========================
# UPLOAD FILE
# =========================
file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Clean
    df = df.drop(columns=["CUST_ID"], errors="ignore")
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Scale + Predict
    scaled = scaler.transform(df)
    clusters = model.predict(scaled)
    df["Cluster"] = clusters

    # =========================
    # SEGMENT LABELS
    # =========================
    labels = {
        0: "💎 VIP Customers",
        1: "💰 High Spenders",
        2: "⚠️ Cash Users",
        3: "❄️ Low Activity"
    }

    df["Segment"] = df["Cluster"].map(labels)

    # =========================
    # LAYOUT
    # =========================
    col1, col2 = st.columns(2)

    # LEFT
    with col1:
        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("📈 Segment Distribution")

        fig, ax = plt.subplots()
        df["Segment"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # RIGHT
    with col2:
        st.subheader("📍 Customer Clusters (PCA)")

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled)

        fig2, ax2 = plt.subplots()
        ax2.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
        ax2.set_xlabel("PCA 1")
        ax2.set_ylabel("PCA 2")
        st.pyplot(fig2)

    # =========================
    # INSIGHTS (SIMPLIFIED)
    # =========================
    st.subheader("📊 Business Insights")

    st.markdown("""
    💎 **VIP Customers** → High balance & frequent usage  
    💰 **High Spenders** → Large purchases  
    ⚠️ **Cash Users** → Depend on cash advance  
    ❄️ **Low Activity** → Rare usage, need re-engagement  
    """)

    # Download button
    st.download_button(
        "📥 Download Segmented Data",
        df.to_csv(index=False),
        file_name="segmented_customers.csv"
    )

else:
    st.info("👆 Upload a CSV file to start segmentation.")