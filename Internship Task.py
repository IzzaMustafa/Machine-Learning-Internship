import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading File

df = pd.read_csv("StudentPerformanceFactors.csv")
print(df)

# Cleaning Data

df = df.dropna()
df = df.drop_duplicates()

# Visualizing Data

plt.figure(figsize=(16,6))
plt.plot(df["Hours_Studied"], df["Exam_Score"], marker = 'o')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Exam Score VS Hours Studied")
plt.show()

# Converting to Numeric Labels

df = pd.get_dummies(df, drop_first= True)

# Splitting Data

X = df.drop(columns= ["Exam_Score"])
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state=42
)

# Linear Regression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Coefficient: ", model.coef_)
print("Intercept: ", model.intercept_)

print("\nLinear Regression")
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("Mean Square Error: ", mean_squared_error(y_test, y_pred))
print("Root Mean Square Error: ", rmse)
print("R² Scores: ", r2_score(y_test, y_pred))

# Polynomial Regression (For Degree 2, Quadratic)

poly_model = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))

print("\nPolynomial Regression:")  # Didn't Improved Performance, Slightly worse than Linear Regression
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred_poly))
print("Mean Square Error: ", mean_squared_error(y_test, y_pred_poly)) 
print("Root Mean Square Error: ", rmse_poly)
print("R² Scores: ", r2_score(y_test, y_pred_poly))

# Experimenting with Different Features

X_new = df[["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Tutoring_Sessions", "Physical_Activity", "Internet_Access_Yes"]]
y_new = df["Exam_Score"]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size= 0.2, random_state= 42
)

model_2 = LinearRegression()
model_2.fit(X_train_new, y_train_new)
y_pred_new = model_2.predict(X_test_new)

print("\nResult with Reduced Features:")
print("R² Scores: ", r2_score(y_test_new, y_pred_new)) # Reduced Accuracy which shows other factors contained useful information