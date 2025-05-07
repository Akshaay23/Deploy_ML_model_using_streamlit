import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open("/workspaces/Deploy_ML_model_using_streamlit/trained_model.sav", 'rb'))

# Prediction function
def diabetes_prediction(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape for a single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    st.title("Diabetes Prediction Web App")

    # Input fields
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Level")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the Person") 

    diagnosis = ''

    if st.button('Diabetes Test Prediction'):
        try:
            # Convert inputs to float and predict
            input_data = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = "Please enter valid numeric values in all fields."

    st.success(diagnosis)

if __name__ == '__main__':
    main()
