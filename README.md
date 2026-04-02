# 🚗 Auto Insurance Churn Prediction App

A deployed Streamlit web application that predicts customer churn risk in the Canadian auto insurance industry using a Logistic Regression model.  

This project goes beyond prediction by providing **explainable insights**, **business recommendations**, and a **downloadable executive report** to support decision-making.

---

## 🔗 Live App

👉 https://auto-insurance-churn-app.streamlit.app  

---

## 📊 Project Overview

Customer churn is a major challenge in the auto insurance industry. This application helps identify customers at risk of leaving and supports proactive retention strategies.

The model is trained on a **synthetic dataset of 10,000 Canadian auto insurance customers**, designed to:
- Avoid privacy concerns  
- Ensure reproducibility  
- Reflect realistic churn patterns  

---

## ⚙️ Features

### 🔹 1. Churn Prediction
- Predicts whether a customer is likely to churn
- Displays churn probability with a progress bar

### 🔹 2. Risk Classification
- Low Risk  
- Moderate Risk  
- High Risk  

### 🔹 3. Business Recommendations
- Provides actionable retention strategies based on risk level

### 🔹 4. Explainable AI (Key Highlight)
- **Model Driver View** → Shows overall feature importance  
- **Customer-Specific Driver View** → Shows what drives THIS customer's risk  

### 🔹 5. Executive PDF Report
- Downloadable customer risk report  
- Includes:
  - Probability
  - Risk level
  - Recommendation
  - Customer inputs  
- Designed for real-world business use  

---

## 🧠 Model Details

- Model: Logistic Regression  
- Preprocessing:
  - Label Encoding for categorical variables  
  - Standard Scaling for numerical features  
- Evaluation Focus:
  - Recall for churn class (important for identifying at-risk customers)

---

## 📈 Key Insights

- Customers with **high premium increases** are more likely to churn  
- **Low satisfaction scores** strongly increase churn risk  
- **Low tenure (new customers)** are at higher risk  
- Loyal customers are significantly less likely to churn  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- ReportLab (for PDF generation)

---

## 📁 Project Structure
