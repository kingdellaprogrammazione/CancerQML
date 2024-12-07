import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

#=====================================================
# Step 1: Load the dataset from a CSV file
#=====================================================
# Replace "your_data.csv" with the actual path to your CSV file
df = pd.read_csv("ex.csv")

# Check if the CSV columns match the expected columns
expected_columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
if not all(col in df.columns for col in expected_columns):
    raise ValueError("CSV file does not contain the expected columns.")

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

#=====================================================
# Step 2: Split into Train/Test sets
#=====================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#=====================================================
# Step 3: Train a Model (XGBoost)
#=====================================================
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

#=====================================================
# Step 4: Analyze Feature Importance
#=====================================================
importances = model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("Feature Importances:")
print(feature_importances)

#=====================================================
# Step 5: Evaluate the model
#=====================================================
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#=====================================================
# Step 6: Use SHAP for Model Explainability (Optional)
#=====================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot (global feature importance)
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.show()

#=====================================================
# Step 7: Generate a Report
#=====================================================
report = f"""
MODEL REPORT
============

Model: XGBClassifier
Train Size: {X_train.shape[0]}
Test Size: {X_test.shape[0]}

Feature Importances:
{feature_importances.to_string()}

Classification Report:
{classification_report(y_test, y_pred)}

Confusion Matrix:
{cm}
"""

print(report)

# Save report to a text file
with open("model_report.txt", "w") as f:
    f.write(report)
