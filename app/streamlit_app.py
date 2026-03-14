import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model

import os
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

model_path = "model/house_price_model.pkl"

# Check if model exists
if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))

else:
    # Train model automatically
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs("model", exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction")
st.write("Enter house details to estimate the price.")

# Sidebar Inputs
st.sidebar.header("Input Features")

MedInc = st.sidebar.slider("Median Income",1.0,15.0,5.0)
HouseAge = st.sidebar.slider("House Age",1,50,30)
AveRooms = st.sidebar.slider("Average Rooms",1.0,10.0,6.0)
AveBedrms = st.sidebar.slider("Average Bedrooms",0.5,5.0,1.0)
Population = st.sidebar.slider("Population",100,5000,1000)
AveOccup = st.sidebar.slider("Average Occupancy",1.0,10.0,3.0)
Latitude = st.sidebar.slider("Latitude",32.0,42.0,34.0)
Longitude = st.sidebar.slider("Longitude",-125.0,-114.0,-118.0)

features = pd.DataFrame({
"MedInc":[MedInc],
"HouseAge":[HouseAge],
"AveRooms":[AveRooms],
"AveBedrms":[AveBedrms],
"Population":[Population],
"AveOccup":[AveOccup],
"Latitude":[Latitude],
"Longitude":[Longitude]
})

# Prediction
if st.button("Predict Price"):

    prediction = model.predict(features)

    st.success(f"💰 Predicted House Value: ${prediction[0]*100000:,.0f}")

# Map Visualization

st.subheader("Location Visualization")

map_df = pd.DataFrame({
    "lat":[Latitude],
    "lon":[Longitude]
})

st.map(map_df)

# Feature Importance
st.subheader("Model Feature Importance")

feature_names = [
"MedInc","HouseAge","AveRooms","AveBedrms",
"Population","AveOccup","Latitude","Longitude"
]

importances = model.feature_importances_

df = pd.DataFrame({
"Feature":feature_names,
"Importance":importances
}).sort_values("Importance",ascending=True)

fig,ax = plt.subplots()
ax.barh(df["Feature"],df["Importance"])
ax.set_title("Feature Importance")
st.pyplot(fig)


# Model Evaluation - Performance Metrics

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df_eval = housing.frame

X_eval = df_eval.drop("MedHouseVal", axis=1)
y_eval = df_eval["MedHouseVal"]

preds = model.predict(X_eval)

r2 = r2_score(y_eval, preds)
mse = mean_squared_error(y_eval, preds)

st.subheader("Model Performance")

col1, col2 = st.columns(2)

col1.metric("R² Score", f"{r2:.2f}")
col2.metric("Mean Squared Error", f"{mse:.2f}")


# Dataset Preview

st.subheader("Dataset Preview")

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df_data = housing.frame

st.dataframe(df_data.head())


# Correlation Heatmap

import seaborn as sns

st.subheader("Feature Correlation")

fig2, ax2 = plt.subplots()
sns.heatmap(df_data.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)