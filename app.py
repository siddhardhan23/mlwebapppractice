# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 14:57:08 2022

@author: siddhardhan.s
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'


# giving a title
st.title('Diabetes Prediction ML Web App')


# getting the input data from the user


col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies')

with col2:
    Glucose = st.text_input('Glucose Level')
    
with col3:
    BloodPressure = st.text_input('Blood Pressure value')
    
with col1:
    SkinThickness = st.text_input('Skin Thickness value')
    
with col2:
    Insulin = st.text_input('Insulin Level')
    
with col3:
    BMI = st.text_input('BMI value')
    
with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
with col2:
    Age = st.text_input('Age of the Person')



# code for Prediction
diagnosis = ''

# creating a button for Prediction

if st.button('Diabetes Test Result'):
    diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    
st.success(diagnosis)
