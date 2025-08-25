# 🏦 Loan Approval Prediction

This project focuses on predicting whether a **loan application** will be **approved or rejected** based on applicant details such as income, CIBIL score, loan amount, and assets.  
By applying **machine learning classification models**, the system aims to help financial institutions make faster, more accurate, and data-driven lending decisions.  

---

## 📌 Objective
- Develop a model that can classify loan applications as **Approved** or **Rejected**.  
- Use applicant features like **income, credit score, loan amount, and assets**.  
- Compare model performance while balancing **accuracy** and **interpretability**.  

---

## ⚙️ Steps Performed

### 🔍 Data Preprocessing
- Removed **extra spaces** from column headers and row entries.  
- Checked for **missing values** (none found, but imputation logic with median/mode was added for robustness).  
- Encoded categorical variables using **One-Hot Encoding (`pd.get_dummies()`)**.  

---

### ⚖️ Handling Class Imbalance
- Original dataset was **imbalanced**:  
  - **2125 Approved** vs **1290 Rejected**  
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.  
- After SMOTE:  
  - **2125 Approved** vs **2125 Rejected** (balanced).  

---

### 🤖 Model Training
I trained and compared two machine learning models:  

1. **Logistic Regression**  
   - Linear, interpretable, serves as a **baseline model**.  

2. **Decision Tree Classifier**  
   - Non-linear, flexible, learns **complex decision rules**.  

---

### 📊 Model Evaluation
Evaluation was done using **classification_report**, measuring:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  

---

## 📊 Results

### Logistic Regression
- Accuracy: **~81%**  
- Precision (Approved): **0.85**  
- Recall (Approved): **0.85**  
- F1-Score (Approved): **0.85**  

### Decision Tree Classifier
- Accuracy: **~97%**  
- Precision (Approved): **0.98**  
- Recall (Approved): **0.97**  
- F1-Score (Approved): **0.98**  

---

## 📝 Insights

- ✅ **Decision Tree** achieved higher accuracy (**97%**) compared to Logistic Regression (**81%**).  
- ⚠️ However, **Decision Trees may overfit** if not pruned or regularized.  
- ✅ **Logistic Regression** offers **interpretability**, which is crucial for financial applications (banks must explain approval/rejection decisions).  

---

### 💡 Precision vs Recall in Loan Prediction
- **Precision** → Fewer **false approvals** → reduces the risk of approving **bad loans**.  
- **Recall** → Fewer **missed approvals** → ensures capturing **all eligible applicants**.  

👉 Thus, the **choice of model depends on business priorities**:  
- If avoiding risky approvals is critical → prioritize **Precision**.  
- If maximizing the number of approved eligible loans is critical → prioritize **Recall**.  

---

## 📂 Dataset & Files
- `loan_approval_dataset.csv` → Input dataset.  
- `Internship Task 4.py` → Python code file.  
- `Internship Task 4.ipynb` → Jupyter notebook (with code + outputs).  

---

## ✨ Conclusion
This project demonstrates the importance of balancing **performance** vs **interpretability** in **machine learning for financial decision-making**.  

- Decision Trees provide **high accuracy** but risk **overfitting**.  
- Logistic Regression offers **explainability**, making it more suitable for **regulatory environments**.  

Financial institutions can choose the model based on whether their priority is **accuracy, risk control, or explainability**.  
