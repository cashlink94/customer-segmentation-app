import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("💳 Customer Segmentation App")
st.caption("AI-powered customer segmentation using KMeans clustering")

st.divider()

# ======================
# SAFE MODEL LOADING
# ======================
try:
    model = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error("❌ Model files not found. Run training first.")
    st.stop()

# ======================
# FILE UPLOAD
# ======================
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:

    df = pd.read_csv(file)

    # ======================
    # SAFE CLEANING
    # ======================
    df = df.drop(columns=["CUST_ID"], errors="ignore")

    # Replace missing values safely
    df = df.fillna(df.median(numeric_only=True))

    # Ensure numeric only
    df = df.select_dtypes(include=["number"])

    if df.shape[1] < 2:
        st.error("❌ Not enough numeric features for clustering.")
        st.stop()

    # ======================
    # SCALING + PREDICTION
    # ======================
    X = scaler.transform(df)
    clusters = model.predict(X)

    df["Cluster"] = clusters

    # ======================
    # SMART LABELING
    # ======================
    label_map = {
        0: "💎 VIP Customers",
        1: "💰 High Spenders",
        2: "⚠️ Cash Users",
        3: "❄️ Low Activity Users"
    }

    df["Segment"] = df["Cluster"].map(label_map)

    # ======================
    # KPI SECTION
    # ======================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", len(df))
    col2.metric("VIP", sum(df["Segment"] == "💎 VIP Customers"))
    col3.metric("High Spenders", sum(df["Segment"] == "💰 High Spenders"))
    col4.metric("Low Activity", sum(df["Segment"] == "❄️ Low Activity Users"))

    st.divider()

    # ======================
    # DATA PREVIEW
    # ======================
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ======================
    # DISTRIBUTION CHART
    # ======================
    st.subheader("📈 Segment Distribution")

    fig, ax = plt.subplots()
    df["Segment"].value_counts().plot(kind="bar", ax=ax)
    ax.set_ylabel("Customers")
    st.pyplot(fig)

    # ======================
    # PCA (FIXED SAFE VERSION)
    # ======================
    st.subheader("📍 Customer Clusters (2D View)")

    try:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X)

        fig2, ax2 = plt.subplots()
        ax2.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=clusters,
            cmap="viridis",
            alpha=0.6
        )

        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        ax2.set_title("Customer Segmentation Clusters")

        st.pyplot(fig2)

    except Exception:
        st.warning("PCA visualization unavailable for this dataset.")

    # ======================
    # BUSINESS INSIGHTS (FINAL CLEAN)
    # ======================
    st.subheader("📊 Business Insights")

    st.markdown("""
### 💎 VIP Customers
High-value users → priority retention + loyalty rewards

### 💰 High Spenders
Frequent large transactions → upsell premium offers

### ⚠️ Cash Users
Cash advance users → monitor credit risk

### ❄️ Low Activity Users
Inactive users → re-engagement campaigns needed
""")

    # ======================
    # DOWNLOAD
    # ======================
    st.download_button(
        "📥 Download Segmented Data",
        df.to_csv(index=False),
        file_name="segmented_customers.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file to start analysis")