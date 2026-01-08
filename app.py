import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
h1 {
    text-align: center;
}
.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    height: 3em;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #ff1f1f;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load artifacts ----------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("churn_model")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    le_gender = pickle.load(open("label_encoder_gender.pkl", "rb"))
    ohe_geo = pickle.load(open("onehot_encoder_geo.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
    return model, scaler, le_gender, ohe_geo, feature_names

model, scaler, le_gender, ohe_geo, feature_names = load_artifacts()

# ---------------- Title ----------------
st.markdown("<h1>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.write("")

# ---------------- Inputs ----------------
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.slider("Age", 18,92)

balance = st.number_input("Balance", min_value=0.0, step=1000.0)
credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, step=1.0)
salary = st.number_input("Estimated Salary", min_value=0.0, step=1000.0)

tenure = st.slider("Tenure (Years)", 0, 10, 0)
num_products = st.slider("Number of Products", 1, 4, 0)

has_cr_card = st.selectbox("Has Credit Card", ["No", "Yes"])
is_active = st.selectbox("Is Active Member", ["No", "Yes"])

# Convert binary inputs
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0

st.write("")

# ---------------- Prediction ----------------
if st.button("Predict Churn"):
    gender_encoded = le_gender.transform([gender])[0]
    geo_encoded = ohe_geo.transform([[geography]])

    geo_df = pd.DataFrame(
        geo_encoded,
        columns=ohe_geo.get_feature_names_out(["Geography"])
    )

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

    input_df = pd.concat([input_df, geo_df], axis=1)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    input_scaled = scaler.transform(input_df)
    prob = model.predict(input_scaled)[0][0]

    st.markdown("---")
    st.subheader("Prediction Result")

    st.metric("Churn Probability", f"{prob:.2%}")

    if prob >= 0.5:
        st.error("🚨 Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

st.markdown("---")
st.caption("🚀 Built with Streamlit & TensorFlow")
