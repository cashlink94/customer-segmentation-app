# 💳 Customer Segmentation App

AI-powered customer segmentation system using **KMeans clustering** and **PCA visualization**.

This project analyzes credit card customer behavior and groups them into meaningful business segments for marketing and decision-making.

---

## 🚀 Live App
👉
[](https://customer-segmentation-app.streamlit.app)

(Replace this after deployment)

---

## 📊 Project Overview

This app segments customers into groups such as:

- 💎 VIP Customers  
- 💰 High Spenders  
- ⚠️ Cash Users  
- ❄️ Low Activity Users  

It helps businesses:
- Improve marketing targeting
- Reduce churn
- Increase customer lifetime value

---

## 🧠 Machine Learning Workflow

### 1. Data Preprocessing
- Removed customer ID column
- Handled missing values
- Scaled numerical features

### 2. Model
- KMeans Clustering
- Optimal cluster selection
- PCA for 2D visualization

### 3. Output
- Customer segmentation labels
- Business insights
- Interactive dashboard

---

## 📂 Project Structure

---

## 📈 Features

✔ Interactive Streamlit dashboard  
✔ Customer clustering using KMeans  
✔ PCA visualization of clusters  
✔ Business insights for each segment  
✔ CSV upload support  
✔ Download segmented data  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Streamlit  
- Joblib  

---

## ▶️ How to Run Locally

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/customer-segmentation-app.git

cd customer-segmentation-app

# Create environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py