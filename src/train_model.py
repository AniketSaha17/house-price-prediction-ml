import pandas as pd
import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Save model
with open("../model/house_price_model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model saved successfully!")