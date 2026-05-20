import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

# Load dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# =====================================================================
# a. Data Preprocessing (Handle missing values, feature scaling)
# =====================================================================
# 1. Explicitly check for missing values (prints 0s, proving we checked)
print("Missing values per column:\n", df.isnull().sum())

# 2. Feature Scaling
# Note: diabetes.data is pre-scaled by sklearn, but we apply StandardScaler 
# here to strictly satisfy the prompt requirement 'a'.
scaler = StandardScaler()

# =====================================================================
# b. Fit a Simple Linear Regression model (Predicting using 'bmi')
# =====================================================================
X_simple = df[['bmi']]
y = df['target']

# Scaling the single feature
X_simple_scaled = scaler.fit_transform(X_simple)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple_scaled, y, test_size=0.2, random_state=42
)

lr_simple = LinearRegression()
lr_simple.fit(X_train_s, y_train_s)
y_pred_s = lr_simple.predict(X_test_s)

# =====================================================================
# c. Extend to Multiple Linear Regression with multiple features
# =====================================================================
X_multi = df.iloc[:, :-1]  # All features except target

# Scaling multiple features
X_multi_scaled = scaler.fit_transform(X_multi)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi_scaled, y, test_size=0.2, random_state=42
)

lr_multi = LinearRegression()
lr_multi.fit(X_train_m, y_train_m)
y_pred_m = lr_multi.predict(X_test_m)

# =====================================================================
# d. Evaluate models using MSE, RMSE, and R² Score
# =====================================================================
print("\n--- Simple Linear Regression Evaluation ---")
print(f"R² Score:  {r2_score(y_test_s, y_pred_s):.4f}")
print(f"MAE:       {mean_absolute_error(y_test_s, y_pred_s):.4f}")
print(f"MSE:       {mean_squared_error(y_test_s, y_pred_s):.4f}")
print(f"RMSE:      {root_mean_squared_error(y_test_s, y_pred_s):.4f}") # Added RMSE

print("\n--- Multiple Linear Regression Evaluation ---")
print(f"R² Score:  {r2_score(y_test_m, y_pred_m):.4f}")
print(f"MAE:       {mean_absolute_error(y_test_m, y_pred_m):.4f}")
print(f"MSE:       {mean_squared_error(y_test_m, y_pred_m):.4f}")
print(f"RMSE:      {root_mean_squared_error(y_test_m, y_pred_m):.4f}")

# =====================================================================
# e. Visualize the regression line and predictions
# =====================================================================
# Plot 1: Simple Linear Regression Line
plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test_s, color='blue', alpha=0.6, label='Actual Data')
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression (BMI vs Diabetes Progression)')
plt.xlabel('Scaled BMI')
plt.ylabel('Disease Progression (Target)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Plot 2: Multiple Linear Regression (Actual vs Predicted)
plt.figure(figsize=(8, 5))
plt.scatter(y_test_m, y_pred_m, color='purple', alpha=0.6, label='Predictions')
plt.plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], color='red', linestyle='--', linewidth=2, label='Perfect Prediction Identity')
plt.title('Multiple Linear Regression (Actual vs Predicted Values)')
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Target Values')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
