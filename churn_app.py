import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

from sklearn.preprocessing import LabelEncoder

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# ---------------------------------
# PDF generation function
# ---------------------------------
def generate_pdf(probability, risk_level, recommendation, customer_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Auto Insurance Customer Churn Report", styles["Title"]))
    content.append(Paragraph("Seneca Business Analytics Capstone Project", styles["Normal"]))
    content.append(Spacer(1, 8))
    content.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Normal"]
        )
    )
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Churn Probability: {probability:.2%}", styles["Normal"]))
    content.append(Paragraph(f"Risk Level: {risk_level}", styles["Normal"]))
    content.append(Paragraph(f"Recommendation: {recommendation}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Customer Profile Summary", styles["Heading2"]))
    content.append(Spacer(1, 8))

    for key, value in customer_data.items():
        content.append(Paragraph(f"{key}: {value}", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer


# ---------------------------------
# Page setup
# ---------------------------------
st.set_page_config(
    page_title="Auto Insurance Churn Prediction App",
    page_icon="📉",
    layout="centered"
)

# ---------------------------------
# Load trained model and scaler
# ---------------------------------
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("churn_scaler.pkl", "rb"))

# ---------------------------------
# Recreate label encoders
# Must match notebook categories
# ---------------------------------
provinces = [
    "Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba",
    "Saskatchewan", "Nova Scotia", "New Brunswick",
    "Newfoundland and Labrador", "Prince Edward Island"
]

age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
income_bands = ["<40k", "40-59k", "60-79k", "80-119k", "120k+"]
urban_rural_options = ["Urban", "Rural"]

encoders = {}

encoders["province"] = LabelEncoder()
encoders["province"].fit(provinces)

encoders["age_group"] = LabelEncoder()
encoders["age_group"].fit(age_groups)

encoders["income_band"] = LabelEncoder()
encoders["income_band"].fit(income_bands)

encoders["urban_rural"] = LabelEncoder()
encoders["urban_rural"].fit(urban_rural_options)

# ---------------------------------
# App title and description
# ---------------------------------
st.title("Auto Insurance Churn Prediction App")
st.subheader("Logistic Regression Prototype Based on Synthetic Canadian Auto Insurance Data")

st.markdown("""
### 📊 About This App
This application predicts customer churn risk in the Canadian auto insurance industry using a Logistic Regression model trained on a synthetic dataset.

It helps identify high-risk customers and supports proactive retention strategies.
""")

st.write(
    "Enter customer details below to estimate churn risk and support proactive retention decisions."
)

# ---------------------------------
# User inputs
# ---------------------------------
province_input = st.selectbox("Province", provinces)
age_group_input = st.selectbox("Age Group", age_groups)
income_band_input = st.selectbox("Income Band", income_bands)
urban_rural_input = st.selectbox("Urban/Rural", urban_rural_options)

years_with_company_input = st.slider("Years with Company", 0, 14, 2)
claim_last_3years_input = st.slider("Claims in Last 3 Years", 0, 5, 0)
premium_increase_pct_input = st.slider("Premium Increase (%)", 0.0, 40.0, 5.0)
satisfaction_input = st.slider("Satisfaction Score", 1, 10, 7)

# Loyalty derived from notebook logic
is_loyal_customer_input = 1 if years_with_company_input >= 3 else 0

st.write(
    f"**Loyal Customer (derived from tenure):** {'Yes' if is_loyal_customer_input == 1 else 'No'}"
)

# ---------------------------------
# Prediction button
# ---------------------------------
if st.button("Predict Churn Risk"):
    input_df = pd.DataFrame({
        "province": [province_input],
        "age_group": [age_group_input],
        "income_band": [income_band_input],
        "urban_rural": [urban_rural_input],
        "years_with_company": [years_with_company_input],
        "claim_last_3years": [claim_last_3years_input],
        "premium_increase_pct": [premium_increase_pct_input],
        "satisfaction": [satisfaction_input],
        "is_loyal_customer": [is_loyal_customer_input]
    })

    # Apply same encoding as notebook
    input_df["province"] = encoders["province"].transform(input_df["province"])
    input_df["age_group"] = encoders["age_group"].transform(input_df["age_group"])
    input_df["income_band"] = encoders["income_band"].transform(input_df["income_band"])
    input_df["urban_rural"] = encoders["urban_rural"].transform(input_df["urban_rural"])

    # Apply same scaling as notebook
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ---------------------------------
    # Results
    # ---------------------------------
    st.markdown("### 📊 Customer Risk Summary")
    st.write("## Prediction Result")

    if prediction == 1:
        st.error("🚨 Churn Predicted")
    else:
        st.success("✅ No Churn Predicted")

    st.write(f"**Churn Probability:** {probability:.2%}")
    st.progress(float(probability))
    st.caption(f"Probability bar: {probability:.2%}")

    # ---------------------------------
    # Risk level and action
    # ---------------------------------
    if probability < 0.30:
        risk = "Low Risk"
        action = "Maintain regular engagement and monitor customer experience."
    elif probability < 0.60:
        risk = "Moderate Risk"
        action = "Review service quality and consider targeted retention support."
    else:
        risk = "High Risk"
        action = "Immediate retention action recommended: premium review, outreach, and service recovery."

    st.write(f"**Risk Level:** {risk}")
    st.write(f"**Suggested Action:** {action}")

    st.markdown("""
### 💡 Business Insight
Customers with high premium increases, low satisfaction, and low loyalty are significantly more likely to churn.

Proactive engagement and pricing strategies can reduce churn risk.
""")

    # ---------------------------------
    # PDF download
    # ---------------------------------
    customer_data = {
        "Province": province_input,
        "Age Group": age_group_input,
        "Income Band": income_band_input,
        "Urban/Rural": urban_rural_input,
        "Years with Company": years_with_company_input,
        "Claims in Last 3 Years": claim_last_3years_input,
        "Premium Increase (%)": premium_increase_pct_input,
        "Satisfaction Score": satisfaction_input,
        "Loyal Customer": "Yes" if is_loyal_customer_input == 1 else "No"
    }

    pdf_file = generate_pdf(probability, risk, action, customer_data)

    st.download_button(
        label="📄 Download Executive Report (PDF)",
        data=pdf_file,
        file_name="customer_churn_report.pdf",
        mime="application/pdf"
    )

    # ---------------------------------
    # Model driver view (static)
    # ---------------------------------
    st.write("## Model Driver View")
    st.caption(
        "Green features reduce churn risk, while red features increase churn risk based on model coefficients."
    )

    feature_names = [
        "province",
        "age_group",
        "income_band",
        "urban_rural",
        "years_with_company",
        "claim_last_3years",
        "premium_increase_pct",
        "satisfaction",
        "is_loyal_customer"
    ]

    coef_values = model.coef_[0]

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coef_values,
        "AbsoluteCoefficient": np.abs(coef_values)
    }).sort_values("AbsoluteCoefficient", ascending=False)

    colors = ["green" if x < 0 else "red" for x in coef_df["Coefficient"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax.invert_yaxis()
    ax.set_title("Key Drivers of Customer Churn")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

    # ---------------------------------
    # Customer-specific driver view (dynamic)
    # ---------------------------------
    st.write("## Customer-Specific Driver View")
    st.caption(
        "This view shows how this specific customer's input values interact with the model coefficients."
    )

    contribution_values = coef_values * input_scaled[0]

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contribution_values,
        "AbsoluteContribution": np.abs(contribution_values)
    }).sort_values("AbsoluteContribution", ascending=False)

    contrib_colors = ["green" if x < 0 else "red" for x in contrib_df["Contribution"]]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.barh(contrib_df["Feature"], contrib_df["Contribution"], color=contrib_colors)
    ax2.invert_yaxis()
    ax2.set_title("Drivers of This Customer's Predicted Risk")
    ax2.set_xlabel("Contribution")
    ax2.set_ylabel("Feature")
    st.pyplot(fig2)

    st.caption(
        "Positive (red) values increase churn risk, while negative (green) values reduce it for this specific customer."
    )

    # ---------------------------------
    # Visual business alert
    # ---------------------------------
    if risk == "High Risk":
        st.warning("⚠️ Immediate attention required: Customer likely to churn.")
    elif risk == "Moderate Risk":
        st.info("ℹ️ Monitor this customer and consider engagement strategies.")
    else:
        st.success("✅ Customer retention is stable.")

# ---------------------------------
# Footer note
# ---------------------------------
st.caption(
    "This prototype is based on a synthetic dataset and is intended for academic demonstration purposes only."
)

st.markdown("""
---
<center>
Developed by <b>Nnamdi Onu</b>, <b>Tsz Yan Chan</b>, <b>Fisayo Adeyinka</b>, <b>Adedoyin Osokoya</b><br>
Seneca Business Analytics Capstone Project
</center>
""", unsafe_allow_html=True)
