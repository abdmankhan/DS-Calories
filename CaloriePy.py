import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import joblib  # For saving and loading the trained model

# Set up Streamlit page
st.set_page_config(page_title="Calorie Prediction App", layout="centered")

# Main heading
st.title("Calorie Prediction App")
st.markdown("### Predict calories burned based on exercise data!")

# Load datasets
@st.cache_data
def load_data():
    calories = pd.read_csv('calories.csv')
    exercise_data = pd.read_csv('exercise.csv')
    # Merge datasets
    calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
    # Replace Gender text with numerical values
    calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)
    return calories_data

calories_data = load_data()

# Display statistics about the dataset
with st.expander("Dataset Overview & Statistics"):
    st.subheader("Dataset Head")
    st.write(calories_data.head())
    st.subheader("Statistics")
    st.write(calories_data.describe())
    st.subheader("Missing Values")
    st.write(calories_data.isnull().sum())

# Split the data into features and target
X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
Y = calories_data['Calories']

# Function to train and save the model
@st.cache_resource
def train_model():
    model = XGBRegressor()
    model.fit(X, Y)
    # Save the trained model
    joblib.dump(model, 'trained_model.pkl')
    return model

# Load pre-trained model (if available) or train a new model
def load_model():
    try:
        # Attempt to load the model from disk
        model = joblib.load('trained_model.pkl')
    except FileNotFoundError:
        # If model is not found, retrain and save it
        model = train_model()
    return model

model = load_model()

# Charts for EDA
with st.expander("Exploratory Data Analysis"):
    st.subheader("Gender Count Plot")
    fig, ax = plt.subplots()
    sns.countplot(x=calories_data['Gender'], ax=ax)
    st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(calories_data['Age'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    correlation = calories_data.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.2f', annot=True, cmap='Blues', ax=ax)
    st.pyplot(fig)

# Prediction form
st.markdown("## Predict Calories Burned")
st.markdown("### Fill in the values below to see the predicted calories burned!")

# Create the input form
with st.form("calorie_form"):
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age (years)", min_value=1, max_value=100, step=1)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=200.0, step=0.1)
    duration = st.number_input("Duration (mins)", min_value=1.0, max_value=500.0, step=1.0)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=200.0, step=0.1)
    body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1)

    # Submit button
    submit = st.form_submit_button("Predict")

# Handle form submission
if submit:
    # Convert gender to numerical value
    gender_num = 0 if gender == "Male" else 1

    # Prepare data for prediction
    input_data = np.array([[gender_num, age, height, weight, duration, heart_rate, body_temp]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Calories Burned: {prediction[0]:.2f}")

# Footer for dataset stats
st.markdown("## Dataset Summary")
st.markdown(
    """
    This app uses a dataset combining exercise information and calories burned to build a predictive model using XGBoost.
    Below are some key insights and visualizations about the data used:
    - **Dataset Size:** 15000 rows, 8 columns.
    - **Target Variable:** Calories burned.
    - **EDA:** Gender distribution, age distribution, correlation heatmap.
    """
)
