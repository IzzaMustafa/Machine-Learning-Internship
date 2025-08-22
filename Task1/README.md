# Student Performance Prediction using Regression Models

## 📌 Project Overview
This project predicts **student exam scores** based on multiple factors such as:
- Hours studied  
- Attendance  
- Sleep hours  
- Previous scores  
- Tutoring sessions  
- Physical activity  
- Internet access  

The goal is to explore how different features affect exam performance and to experiment with different regression models.

---

## 🛠️ Libraries Used
- pandas  
- numpy  
- matplotlib  
- scikit-learn  

---

## 📂 Dataset
- **File:** `StudentPerformanceFactors.csv` 

---

## 📊 Models Implemented
### 1. Linear Regression
- **R² Score:** `0.7314`  
- **RMSE:** `2.04`  

### 2. Polynomial Regression (Degree 2)
- **R² Score:** `0.7027`  
- **RMSE:** `2.14`  
➡️ Polynomial regression performed **slightly worse** than linear regression.

### 3. Experiment with Selected Features
- Kept features: `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`, `Internet_Access_Yes`  
- **R² Score:** `0.6149`  
➡️ Performance dropped, showing that removing other features reduces prediction accuracy.

---

## 📊 Results
- Linear Regression: Best performance (R² = 0.73)  
- Polynomial Regression: Slightly worse (R² = 0.70)  
- Feature Experimentation: Accuracy dropped (R² = 0.61)  

## ✨ Bonus Task
- Implemented Polynomial Regression  
- Experimented with different feature sets  

## 👩‍💻 Author
**Izza Mustafa Jadoon**  
Machine Learning Internship Project
