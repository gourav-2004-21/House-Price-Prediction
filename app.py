import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/model.pkl")

st.set_page_config(page_title="House Price Prediction")

st.title("🏠 House Price Prediction System")

st.write("Enter House Details")

# Inputs
area = st.number_input("Area (sq ft)", min_value=500)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

stories = st.number_input("Stories", min_value=1)

parking = st.number_input("Parking", min_value=0)

mainroad = st.selectbox("Main Road", ["Yes", "No"])

guestroom = st.selectbox("Guest Room", ["Yes", "No"])

basement = st.selectbox("Basement", ["Yes", "No"])

hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])

airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])

prefarea = st.selectbox("Preferred Area", ["Yes", "No"])

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

# Encoding
yes_no = {
    "Yes": 1,
    "No": 0
}

furnishing_map = {
    "Furnished": 0,
    "Semi-Furnished": 1,
    "Unfurnished": 2
}

# Input array
input_data = np.array([[
    area,
    bedrooms,
    bathrooms,
    stories,
    yes_no[mainroad],
    yes_no[guestroom],
    yes_no[basement],
    yes_no[hotwaterheating],
    yes_no[airconditioning],
    parking,
    yes_no[prefarea],
    furnishing_map[furnishingstatus]
]])

# Prediction
if st.button("Predict Price"):

    prediction = model.predict(input_data)

    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")