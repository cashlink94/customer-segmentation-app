import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# =========================
# Title
# =========================
st.title("💳 Customer Segmentation Dashboard")
st.markdown("AI-powered customer segmentation using KMeans clustering.")

# =========================
# Load Model
# =========================
try:
    model = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except:
    st.error("❌ Model files not found. Please train the model first.")
    st.stop()

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # =========================
    # Clean Data
    # =========================
    df = df.drop(columns=["CUST_ID"], errors="ignore")
    df.fillna(df.mean(), inplace=True)

    # =========================
    # Scale + Predict
    # =========================
    scaled_data = scaler.transform(df)
    df["Cluster"] = model.predict(scaled_data)

    # =========================
    # Segment Labels
    # =========================
    segment_names = {
        0: "💎 VIP Customers",
        1: "💰 High Spenders",
        2: "⚠️ Cash Advance Users",
        3: "❄️ Low Activity Users"
    }

    df["Segment"] = df["Cluster"].map(segment_names)

    # =========================
    # Layout (2 Columns)
    # =========================
    col1, col2 = st.columns(2)

    # -------- LEFT SIDE --------
    with col1:

        with st.expander("📊 Raw Data"):
            st.dataframe(df.head())

        st.subheader("📈 Segment Distribution")
        fig, ax = plt.subplots()
        df["Segment"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Customers")
        plt.xticks(rotation=20)
        st.pyplot(fig)

    # -------- RIGHT SIDE --------
    with col2:

        st.subheader("📍 Customer Clusters (PCA)")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(
            pca_data[:, 0],
            pca_data[:, 1],
            c=df["Cluster"]
        )
        ax2.set_xlabel("PCA 1")
        ax2.set_ylabel("PCA 2")
        st.pyplot(fig2)

    # =========================
    # Insights Section
    # =========================
    st.subheader("📊 Business Insights")

    st.markdown("""
    💎 **VIP Customers**  
    High balance, frequent purchases → target with premium offers  

    💰 **High Spenders**  
    Large transactions → upsell high-value products  

    ⚠️ **Cash Advance Users**  
    Frequent withdrawals → risk monitoring needed  

    ❄️ **Low Activity Users**  
    Minimal usage → re-engagement campaigns  
    """)

else:
    st.info("👆 Upload a dataset to begin")