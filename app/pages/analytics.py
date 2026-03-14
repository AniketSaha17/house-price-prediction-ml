import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Model Analytics")

model = pickle.load(open("model/house_price_model.pkl","rb"))

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