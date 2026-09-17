import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('knn_model.pkl')

st.title("Iris Species Prediction")

st.write("Enter the measurements for the Iris flower:")

sepal_length = st.number_input("SepalLengthCm", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("SepalWidthCm", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("PetalLengthCm", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("PetalWidthCm", min_value=0.0, max_value=10.0, value=1.5)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"The predicted Iris species is: {prediction[0]}")
