# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset (Replace 'data.csv' with your dataset file)
data = pd.read_csv('data.csv')

# Data preprocessing
# Drop any rows with missing values
data.dropna(inplace=True)

# Separate features and target variable
X = data.drop('price', axis=1)  # Features
y = data['price']  # Target variable

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but often recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection and training
# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Random Forest Regressor Model (you can try other models as well)
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

linear_mse = evaluate_model(linear_model, X_test_scaled, y_test)
forest_mse = evaluate_model(forest_model, X_test_scaled, y_test)

print("Linear Regression Model Mean Squared Error:", linear_mse)
print("Random Forest Regressor Model Mean Squared Error:", forest_mse)

# Optional: Cross-validation for better evaluation
linear_cv_scores = cross_val_score(linear_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
forest_cv_scores = cross_val_score(forest_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

print("Linear Regression Cross-Validation Mean Squared Error:", -linear_cv_scores.mean())
print("Random Forest Regressor Cross-Validation Mean Squared Error:", -forest_cv_scores.mean())

# Now you can use these models to predict house prices for new data
