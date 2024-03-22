import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# app heading
st.write("""
# Wine Quality Prediction 
This app predicts the **Wine Quality** type!
""")

# creating sidebar for user input features
st.sidebar.header('Input Parameters')

def user_input_features():
    fixed_acidity = st.sidebar.slider('fixed acidity', 4.6, 15.9, 8.31)
    volatile_acidity = st.sidebar.slider('volatile acidity', 0.12, 1.58, 0.52)
    citric_acid = st.sidebar.slider('citric acid', 0.0, 1.0, 0.5)
    residual_sugar = st.sidebar.slider('residual sugar', 0.9, 15.5, 2.5)
    chlorides = st.sidebar.slider('chlorides', 0.01, 0.6, 0.08)
    free_sulfur_dioxide = st.sidebar.slider('free sulfur dioxide', 1.0, 72.0, 15.0)
    total_sulfur_dioxide = st.sidebar.slider('total sulfur dioxide', 6.0, 289.0, 46.0)
    density = st.sidebar.slider('density', 0.99, 1.04, 0.995)
    pH = st.sidebar.slider('pH', 2.7, 4.0, 3.3)
    sulphates = st.sidebar.slider('sulphates', 0.33, 2.0, 0.65)
    alcohol = st.sidebar.slider('alcohol', 8.4, 14.9, 10.4)

    data = {'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('Input parameters')
st.write(df)

# reading csv file
with open("winequality-red.csv", "r") as f:
    data = [line.strip().split(';') for line in f.readlines()]

column_names = data[0]
data = data[1:]
data = pd.DataFrame(data, columns=column_names)
data.columns = [col.replace('"', '') for col in data.columns]  # Remove double quotes from column names
data = data.astype(float)

# create X and Y arrays
X = np.array(data.iloc[:, :-1].values)  # select all rows and all columns except the last one
Y = np.array(data.iloc[:, -1].values)  # select all rows and the last column (target variable)

# random forest model
wine_quality_rfr = RandomForestRegressor()
wine_quality_rfr.fit(X, Y)

prediction = wine_quality_rfr.predict(df)

st.subheader('Wine Quality')
# st.subheader('Player Salary')
st.subheader(str(np.round(prediction[0], 2)))
# st.write(prediction)