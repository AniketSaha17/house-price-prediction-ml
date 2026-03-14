import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

st.title("📚 Dataset Explorer")

housing = fetch_california_housing(as_frame=True)
df = housing.frame

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig)