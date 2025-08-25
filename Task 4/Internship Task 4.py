import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load File
df = pd.read_csv("loan_approval_dataset.csv")

# Remove White spaces from Column Headers and Row Entries

df.columns = df.columns.str.strip()
df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# Check for Missing Values in Dataset

print("Missing values in dataset:")
print(df.isnull().sum())

# Sample Handling Missing Values (Since No Missing Value)

df['loan_amount'] = df['loan_amount'].fillna(df['loan_amount'].median())
df['education'] = df['education'].fillna(df['education'].mode()[0])

# Encode Target Variable

df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

# Encode Categorical Features into dummy variables

df = pd.get_dummies(df, drop_first= True)

# Split Features & Target Variable

X = df.drop("loan_status", axis= 1)
y = df["loan_status"]

# Train-test split (with stratification for balanced distribution)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state= 42, stratify= y
)

# Handle Imbalance with SMOTE (oversampling minority class)

smote = SMOTE(random_state= 42)
X_train_re, y_train_re = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:\n", y_train.value_counts())
print("\nAfter SMOTE:\n", y_train_re.value_counts())

# Train Logistic Regression Model

log_reg = LogisticRegression(max_iter= 1000, random_state= 42)
log_reg = log_reg.fit(X_train_re, y_train_re)

# Train Decision Tree Model

tree = DecisionTreeClassifier(random_state= 42)
tree = tree.fit(X_train_re, y_train_re)

# Make Predictions

y_pred_log = log_reg.predict(X_test)
y_pred_tree = tree.predict(X_test)

# Evaluate Models

print("\nLogistic Regression: \n", classification_report(y_test, y_pred_log))
print("Decision Tree: \n", classification_report(y_test, y_pred_tree))