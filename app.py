import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Fintech Customer Segmentation", layout="wide")

# =========================
# FINTECH UI (CSS STYLE)
# =========================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stMetric {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("💳 Fintech Customer Segmentation Dashboard")
st.markdown("AI-powered customer intelligence using KMeans clustering")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    kmeans = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return kmeans, scaler

model, scaler = load_model()

# =========================
# UPLOAD DATA
# =========================
file = st.file_uploader("Upload Customer CSV", type=["csv"])

if file:

    df = pd.read_csv(file)
    df = df.drop(columns=["CUST_ID"], errors="ignore")
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # =========================
    # PREDICT CLUSTERS
    # =========================
    scaled = scaler.transform(df)
    clusters = model.predict(scaled)
    df["Cluster"] = clusters

    # =========================
    # CLUSTER LABELS
    # =========================
    labels = {
        0: "💎 VIP Customers",
        1: "💰 High Spenders",
        2: "⚠️ Cash Users",
        3: "❄️ Low Activity Users"
    }

    df["Segment"] = df["Cluster"].map(labels)

    # =========================
    # KPI METRICS (FINTECH STYLE)
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", len(df))
    col2.metric("VIP Customers", sum(df["Segment"] == "💎 VIP Customers"))
    col3.metric("High Spenders", sum(df["Segment"] == "💰 High Spenders"))
    col4.metric("Low Activity", sum(df["Segment"] == "❄️ Low Activity Users"))

    st.divider()

    # =========================
    # RAW DATA
    # =========================
    st.subheader("📊 Customer Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # =========================
    # DISTRIBUTION CHART
    # =========================
    st.subheader("📈 Segment Distribution")

    fig, ax = plt.subplots()
    df["Segment"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # =========================
    # PCA VISUALIZATION
    # =========================
    st.subheader("📍 Customer Segments (PCA View)")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)

    fig2, ax2 = plt.subplots()
    ax2.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap="viridis")
    st.pyplot(fig2)

    # =========================
    # ADVANCED CLUSTER ANALYSIS
    # =========================
    st.subheader("📊 Cluster Intelligence Report")

    cluster_summary = df.groupby("Segment").mean(numeric_only=True)

    st.dataframe(cluster_summary, use_container_width=True)

    # =========================
    # BUSINESS INSIGHTS (UPGRADED)
    # =========================
    st.subheader("💡 Business Insights")

    st.markdown("""
### 💎 VIP Customers
- Highest credit balance and transaction frequency  
- Ideal for premium offers and loyalty programs  

### 💰 High Spenders
- Strong purchase volume  
- Good candidates for credit limit increase  

### ⚠️ Cash Users
- Heavy cash advance dependency  
- Higher financial risk profile  

### ❄️ Low Activity Users
- Minimal engagement  
- Reactivation campaigns recommended  
""")

    # =========================
    # DOWNLOAD
    # =========================
    st.download_button(
        "📥 Download Segmented Data",
        df.to_csv(index=False),
        file_name="segmented_customers.csv"
    )

else:
    st.info("Upload a CSV file to start analysis")