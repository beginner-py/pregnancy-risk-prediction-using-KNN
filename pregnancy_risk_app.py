#!/usr/bin/env python
# coding: utf-8

# In[35]:


import streamlit as st
import pickle
import numpy as np


# In[38]:


# Load model
with open('prediction.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl','rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Streamlit App Title
st.title("Maternal Health and Pregnancy Risk Prediction")

st.write("### Enter Health Details Below:")


# In[39]:


# Input Fields
Age = st.number_input("Age:", min_value=0)
SBP = st.number_input("Systolic Blood Pressure (SBP):", min_value=0)
DBP = st.number_input("Diastolic Blood Pressure (DBP):", min_value=0)
Blood_sugar_HbA1c = st.number_input("HbA1c Value:", min_value=0.0)
Body_Temp = st.number_input("Body Temperature (°F):", min_value=0.0)
BMI = st.number_input("BMI:", min_value=0.0)

# Toggle Inputs for Binary Variables (Yes = 1, No = 0)
previous_complication = 1 if st.toggle("Previous Complication?") else 0
preexisting_diabetes = 1 if st.toggle("Pre-existing Diabetes?") else 0
Gestational_diabetes = 1 if st.toggle("Gestational Diabetes?") else 0
mental_health = 1 if st.toggle("Mental Health Issue?") else 0

Heart_rate = st.number_input("Heart Rate:", min_value=0)


# In[41]:


if st.button("Predict"):
    features = np.array([[Age, SBP, DBP, Blood_sugar_HbA1c, Body_Temp, BMI,
                          previous_complication, preexisting_diabetes,
                          Gestational_diabetes, mental_health, Heart_rate]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

     # Output result
    risk = "High Risk" if prediction[0] == 0 else "Low Risk"
    st.success(f"Predicted Risk Level: {risk}")


# In[ ]:





# In[ ]:





# In[ ]:




