# ============================================
# 1. Import Required Libraries
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ============================================
# 2. Load Dataset
# ============================================

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# ============================================
# 3. Exploratory Data Analysis (EDA)
# ============================================

# Histogram
df.hist(figsize=(12,10))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(df["MedInc"], df["MedHouseVal"])
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Income vs House Price")
plt.show()


# ============================================
# 4. Separate Features and Target
# ============================================

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================
# 5. Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Set:", X_train.shape)
print("Testing Set:", X_test.shape)


# ============================================
# 6. Linear Regression Model
# ============================================

lr_model = LinearRegression()

# Train
lr_model.fit(X_train, y_train)

# Predict
y_pred = lr_model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Evaluation")
print("MSE:", mse)
print("R2 Score:", r2)


# ============================================
# 7. Feature Scaling
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_scaled = LinearRegression()

lr_scaled.fit(X_train_scaled, y_train)

scaled_pred = lr_scaled.predict(X_test_scaled)

print("\nLinear Regression After Scaling")
print("MSE:", mean_squared_error(y_test, scaled_pred))
print("R2:", r2_score(y_test, scaled_pred))


# ============================================
# 8. Random Forest Model
# ============================================

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Evaluation")
print("MSE:", mean_squared_error(y_test, rf_pred))
print("R2:", r2_score(y_test, rf_pred))


# ============================================
# 9. Single House Prediction
# ============================================

sample_house = X_test.iloc[[0]]

rf_single_pred = rf_model.predict(sample_house)

print("\nSingle House Prediction")
print("Predicted:", rf_single_pred[0])
print("Actual:", y_test.iloc[0])


# ============================================
# 10. Feature Importance
# ============================================

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

feature_importance.sort_values().plot(
    kind="barh",
    figsize=(8,6)
)

plt.title("Feature Importance")
plt.show()


# ============================================
# 11. Hyperparameter Tuning
# ============================================

param_grid = {
    "n_estimators":[50,100,200],
    "max_depth":[None,10,20],
    "min_samples_split":[2,5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="r2"
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)


# Evaluate Best Model
best_pred = best_model.predict(X_test)

print("\nBest Model Evaluation")
print("MSE:", mean_squared_error(y_test, best_pred))
print("R2:", r2_score(y_test, best_pred))


# ============================================
# 12. Save Model
# ============================================

with open("best_rf_model.pkl", "wb") as f:
    pickle.dump(best_model, f)


# ============================================
# 13. Load Model
# ============================================

with open("best_rf_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

loaded_pred = loaded_model.predict(X_test)

print("\nLoaded Model Evaluation")
print("MSE:", mean_squared_error(y_test, loaded_pred))
print("R2:", r2_score(y_test, loaded_pred))



# The code above demonstrates the complete workflow of loading the California housing dataset, performing exploratory data analysis, training a linear regression model, evaluating it, and then training a random forest model with hyperparameter tuning. Finally, it saves the best model using pickle for future use.
# Note: In a real-world scenario, you would typically separate this code into different functions or scripts for better organization and maintainability.
# Additionally, you might want to handle missing values, outliers, and perform more advanced feature engineering for improved model performance.
# This code is meant for educational purposes to illustrate the process of building a machine learning model from start to finish.
# Always remember to evaluate your models on unseen data and consider using cross-validation for more robust performance estimates.
# Happy coding!


