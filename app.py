import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("💳 Customer Segmentation App")
st.write("Upload customer data to predict segments using KMeans clustering.")

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------------- DATA PREVIEW ----------------
    with st.expander("📊 Raw Data Preview"):
        st.dataframe(df.head(50), use_container_width=True)

    # ---------------- PREPROCESS ----------------
    df_clean = df.drop("CUST_ID", axis=1, errors="ignore")
    df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))

    # ---------------- SCALE + PREDICT ----------------
    X_scaled = scaler.transform(df_clean)
    clusters = model.predict(X_scaled)

    df["Cluster"] = clusters

    # ---------------- LABELS ----------------
    cluster_labels = {
        0: "💎 VIP",
        1: "💰 Spenders",
        2: "⚠️ Cash Users",
        3: "❄️ Low Activity"
    }

    df["Segment"] = df["Cluster"].map(cluster_labels)

    # ---------------- TOP SECTION (2 COLS) ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Segmented Customers")
        st.dataframe(df[["Cluster", "Segment"]].head(50), use_container_width=True)

    with col2:
        st.subheader("📈 Segment Distribution")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        df["Segment"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    # ---------------- PCA + INSIGHTS ----------------
    st.subheader("📍 Customer Clusters (PCA)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    col3, col4 = st.columns([2, 1])

    with col3:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        scatter = ax2.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=clusters,
            cmap="viridis",
            s=5
        )
        ax2.set_title("Clusters (PCA)")
        ax2.set_xlabel("PCA 1")
        ax2.set_ylabel("PCA 2")
        st.pyplot(fig2)

    with col4:
        st.subheader("📊 Insights")
        st.markdown("""
        💎 **VIP**  
        High balance, frequent usage  

        💰 **Spenders**  
        Large purchases  

        ⚠️ **Cash Users**  
        Frequent withdrawals  

        ❄️ **Low Activity**  
        Minimal usage  
        """)
