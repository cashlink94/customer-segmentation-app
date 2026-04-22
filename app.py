import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Intelligence SaaS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# SIDEBAR - THEME SWITCH
# =========================
theme = st.sidebar.radio("🎨 Theme", ["Dark", "Light"])

if theme == "Dark":
    bg = "#0f172a"
    text = "white"
    card = "#1e293b"
else:
    bg = "#f8fafc"
    text = "#0f172a"
    card = "#e2e8f0"

st.markdown(f"""
<style>
.main {{
    background-color: {bg};
    color: {text};
}}
div[data-testid="stMetric"] {{
    background-color: {card};
    padding: 15px;
    border-radius: 12px;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("💳 Customer Intelligence SaaS Platform")
st.caption("AI-powered segmentation + behavioral analytics")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return (
        joblib.load("models/kmeans_model.pkl"),
        joblib.load("models/scaler.pkl")
    )

model, scaler = load_model()

# =========================
# UPLOAD DATA
# =========================
file = st.file_uploader("Upload Customer Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)
    df = df.drop(columns=["CUST_ID"], errors="ignore")
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # =========================
    # PREDICTION
    # =========================
    X = scaler.transform(df)
    clusters = model.predict(X)
    df["Cluster"] = clusters

    # =========================
    # SMART LABELS
    # =========================
    labels = {
        0: "💎 VIP",
        1: "💰 High Spenders",
        2: "⚠️ Cash Users",
        3: "❄️ Low Activity"
    }

    df["Segment"] = df["Cluster"].map(labels)

    # =========================
    # KPI CARDS
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", len(df))
    col2.metric("VIP", sum(df["Segment"] == "💎 VIP"))
    col3.metric("High Spenders", sum(df["Segment"] == "💰 High Spenders"))
    col4.metric("Inactive", sum(df["Segment"] == "❄️ Low Activity"))

    st.divider()

    # =========================
    # DATA PREVIEW
    # =========================
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # =========================
    # SEGMENT DISTRIBUTION
    # =========================
    st.subheader("📈 Segment Distribution")

    fig, ax = plt.subplots()
    df["Segment"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # =========================
    # PCA VISUALIZATION
    # =========================
    st.subheader("📍 Customer Segments Map (PCA)")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    fig2, ax2 = plt.subplots()
    ax2.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap="viridis")
    st.pyplot(fig2)

    # =========================
    # ADVANCED CLUSTER INTELLIGENCE
    # =========================
    st.subheader("🧠 Cluster Behavioral Intelligence")

    cluster_profile = df.groupby("Segment").mean(numeric_only=True)
    st.dataframe(cluster_profile, use_container_width=True)

    # =========================
    # BUSINESS EXPLANATION (IMPROVED)
    # =========================
    st.subheader("💡 AI Business Insights Engine")

    st.markdown("""
### 💎 VIP Customers
- High balance + high transaction frequency  
- Strong retention value → **target premium offers**

### 💰 High Spenders
- High purchase volume  
- Good credit expansion candidates  

### ⚠️ Cash Users
- Depend heavily on cash advances  
- Higher risk → monitor closely  

### ❄️ Low Activity
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
    st.info("Upload dataset to start analysis")