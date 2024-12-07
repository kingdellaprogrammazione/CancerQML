import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

#=====================================================
# Step 0: Load the dataset
#=====================================================
df = pd.read_csv("ex.csv")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

#=====================================================
# Step 1: Evaluate the model BEFORE removing features and imputing zeros
#=====================================================
X_train_bf, X_test_bf, y_train_bf, y_test_bf = train_test_split(X, y, test_size=0.2, random_state=42)
model_before = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_before.fit(X_train_bf, y_train_bf)
y_pred_bf = model_before.predict(X_test_bf)
acc_before = accuracy_score(y_test_bf, y_pred_bf)

print("Performance BEFORE modifications:")
print("Classification Report:")
print(classification_report(y_test_bf, y_pred_bf))
print("Confusion Matrix:")
print(confusion_matrix(y_test_bf, y_pred_bf))
print(f"Accuracy Before: {acc_before:.4f}")

#=====================================================
# Step 2: Remove the specified features
#=====================================================
features_to_remove = ["Insulin", "SkinThickness", "Pregnancies"]
X_modified = X.drop(columns=features_to_remove, errors='ignore')  # `errors='ignore'` in case these columns are already absent

#=====================================================
# Step 3: Impute zeros using KNN for remaining problematic columns
# Typically, these are columns where zero is not valid: Glucose, BloodPressure, BMI
# Check which of these columns exist after removal
columns_to_impute = [col for col in ["Glucose", "BloodPressure", "BMI"] if col in X_modified.columns]

# Replace zeros with NaN
X_imputed = X_modified.copy()
X_imputed[columns_to_impute] = X_imputed[columns_to_impute].replace(0, np.nan)

# KNN Imputation
imputer = KNNImputer(n_neighbors=5)
X_imputed[columns_to_impute] = imputer.fit_transform(X_imputed[columns_to_impute])

#=====================================================
# Step 4: Evaluate the model AFTER modifications
#=====================================================
X_train_af, X_test_af, y_train_af, y_test_af = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model_after = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_after.fit(X_train_af, y_train_af)
y_pred_af = model_after.predict(X_test_af)
acc_after = accuracy_score(y_test_af, y_pred_af)

print("\nPerformance AFTER removing features and imputing zeros:")
print("Classification Report:")
print(classification_report(y_test_af, y_pred_af))
print("Confusion Matrix:")
print(confusion_matrix(y_test_af, y_pred_af))
print(f"Accuracy After: {acc_after:.4f}")

#=====================================================
# Step 5: Plot the difference in accuracy
#=====================================================
plt.figure(figsize=(6,4))
plt.bar(["Before", "After"], [acc_before, acc_after], color=["red","green"])
plt.title("Accuracy Before vs After Modifications")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
