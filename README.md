# 🏠 House Price Prediction (Warm-up ML Project)


## Objective
Build a regression model to predict median house values using the California Housing dataset.

## Model
Random Forest Regressor

## Features
Median Income
House Age
Average Rooms
Average Bedrooms
Population
Average Occupancy
Latitude
Longitude

## Steps Performed
- Loaded dataset using sklearn
- Performed feature-target separation
- Train-test split (80/20)
- Applied Linear Regression
- Applied Random Forest Regressor
- Evaluated using MSE and R²

## Results

| Model | R² Score |
|-------|----------|
| Linear Regression | 0.57 |
| Random Forest | 0.80 |

## Conclusion
Random Forest significantly outperformed Linear Regression, indicating non-linear relationships in the dataset.



## ▶️ Run the project locally

### 1️⃣ Install dependencies
```
pip install -r requirements.txt
```
### 2️⃣ Train the model
```
python src/train_model.py
```
### 3️⃣ Run the Streamlit app
```
streamlit run app/streamlit_app.py
```