### 🩺 Maternal Health and Pregnancy Risk Prediction App

This project is an end-to-end Pregnancy Risk Level Prediction System developed using Supervised Machine Learning (K-Nearest Neighbors) and deployed as a simple interactive web app using Streamlit.
It predicts whether a pregnant individual is at High Risk or Low Risk based on clinical health parameters.

Dataset
The dataset is open-source, obtained from Kaggle, containing clinical data of pregnant individuals.

Features include:
Age, Systolic & Diastolic Blood Pressure (SBP, DBP), HbA1c (Blood Sugar), Body Temperature, BMI, History of Previous Complications, Pre-existing Diabetes, Gestational Diabetes, Mental Health Issues, Heart Rate, Risk Label (Target: High Risk / Low Risk)

Model Overview
Model Used: K-Nearest Neighbors (KNN)
Accuracy: ~98% on test data
Scaler: StandardScaler to normalize clinical inputs
Target Output: High Risk (1) or Low Risk (0)

Streamlit Web App
An interactive local web application was developed using Streamlit to allow users to input health parameters and receive a predicted risk level instantly.

<img width="1855" height="798" alt="image" src="https://github.com/user-attachments/assets/1856b486-83c3-4445-84cd-db1d7aa750d0" />
<img width="1749" height="623" alt="image" src="https://github.com/user-attachments/assets/656dbcbd-380d-47f5-9e38-8b850ce6120e" />

