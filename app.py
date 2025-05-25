import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title
st.title("Heart Disease Prediction Web App")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

heart_data = load_data()

# Data preprocessing
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy display
st.subheader("Model Evaluation")
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

st.write(f"Training Accuracy: {train_accuracy:.2f}")
st.write(f"Test Accuracy: {test_accuracy:.2f}")

# Sidebar inputs
st.sidebar.header("Input Patient Data")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.sidebar.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    cp = st.sidebar.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", value=120)
    chol = st.sidebar.number_input("Cholesterol", value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 (1 = True; 0 = False)", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes; 0 = No)", [0, 1])
    oldpeak = st.sidebar.number_input("ST depression", value=1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    ca = st.sidebar.selectbox("Major Vessels Colored (0–4)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3])
    
    features = (age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal)
    return np.asarray(features).reshape(1, -1)

input_data = user_input_features()

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("The person **has heart disease**.")
    else:
        st.success("The person **does NOT have heart disease**.")
