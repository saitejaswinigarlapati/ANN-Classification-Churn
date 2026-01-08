import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Prediction")
st.write("Predict whether a customer is likely to **churn or stay**.")

# ----------------------------
# Load model & preprocessors
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("churn_model")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    le_gender = pickle.load(open("label_encoder_gender.pkl", "rb"))
    ohe_geo = pickle.load(open("onehot_encoder_geo.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
    return model, scaler, le_gender, ohe_geo, feature_names

model, scaler, le_gender, ohe_geo, feature_names = load_artifacts()

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("🧾 Customer Details")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", 18, 100, 35)
tenure = st.sidebar.number_input("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance", 0.0, 300000.0, 50000.0)
num_products = st.sidebar.number_input("Number of Products", 1, 4, 2)
has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 300000.0, 60000.0)

# ----------------------------
# Predict
# ----------------------------
if st.button("🔍 Predict Churn"):
    # Encode gender
    gender_encoded = le_gender.transform([gender])[0]

    # Encode geography
    geo_encoded = ohe_geo.transform([[geography]])
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=ohe_geo.get_feature_names_out(["Geography"])
    )

    # Create input dataframe (base features)
    input_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Gender": gender_encoded,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary
    }])

    # Combine with geo features
    input_df = pd.concat([input_df, geo_df], axis=1)

    # Ensure correct feature order
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict(input_scaled)[0][0]

    st.subheader("📊 Prediction Result")
    st.write(f"**Churn Probability:** `{prob:.2f}`")

    if prob >= 0.5:
        st.error("🚨 Customer is **LIKELY TO CHURN**")
    else:
        st.success("✅ Customer is **LIKELY TO STAY**")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TensorFlow")
