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
# Step 1: Detect the importance of factors
#=====================================================
X_train_init, X_test_init, y_train_init, y_test_init = train_test_split(X, y, test_size=0.2, random_state=42)
model_init = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_init.fit(X_train_init, y_train_init)

importances = model_init.feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("Initial Feature Importances:")
print(feature_importances)

# Plot initial feature importances
plt.figure(figsize=(8,6))
feature_importances.plot(kind='bar', title='Initial Feature Importances')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

#=====================================================
# Step 2: Show the amount of zeros
#=====================================================
zero_counts = (X == 0).sum()
zero_percentage = (X == 0).mean() * 100
print("\nCount of Zeros in Each Feature:")
print(zero_counts)
print("\nPercentage of Zeros in Each Feature:")
print(zero_percentage)

#=====================================================
# Step 3: Sanitize the dataset
# Remove at least 3 features that are both low in importance and have lots of zeros.
# We'll remove:
# - The bottom 30% of features by importance.
# - Among them, pick those that have more than 40% zeros.
# - Ensure at least 3 features are removed.

# Define thresholds
importance_cutoff = int(len(feature_importances) * 0.7)  # top 70% are kept
zero_threshold = 40.0

least_important_features = feature_importances.index[importance_cutoff:]  # bottom 30%
candidates_to_remove = zero_percentage[zero_percentage > zero_threshold].index
features_to_remove = list(set(least_important_features) & set(candidates_to_remove))

# If fewer than 3 features match this criterion, remove additional bottom features to ensure at least 3 are removed
if len(features_to_remove) < 3:
    bottom_features = list(feature_importances.index[-3:])  # last 3 most unimportant features
    for f in bottom_features:
        if f not in features_to_remove:
            features_to_remove.append(f)
features_to_remove = features_to_remove[:3]

print("\nFeatures Removed Due to Low Importance and Many Zeros:")
print(features_to_remove)

X_sanitized = X.drop(columns=features_to_remove)

#=====================================================
# Step 4: For the rest of zeros, use KNN to fill those in
# Identify columns typically imputed (no zero values expected)
columns_to_impute = [col for col in X_sanitized.columns if col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]]

X_imputed = X_sanitized.copy()
X_imputed[columns_to_impute] = X_imputed[columns_to_impute].replace(0, np.nan)

imputer = KNNImputer(n_neighbors=5)
X_imputed[columns_to_impute] = imputer.fit_transform(X_imputed[columns_to_impute])

#=====================================================
# Step 5: Before-After Comparison
#=====================================================
def train_and_evaluate(X_data, y_data, desc):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{desc} - Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{desc} - Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    return model, acc

print("\nPerformance BEFORE sanitation:")
model_before, acc_before = train_and_evaluate(X, y, "Before Sanitation")

print("\nPerformance AFTER sanitation:")
model_after, acc_after = train_and_evaluate(X_imputed, y, "After Sanitation")

#=====================================================
# Plot the difference in accuracy
#=====================================================
plt.figure(figsize=(6,4))
plt.bar(["Before", "After"], [acc_before, acc_after], color=["red","green"])
plt.title("Accuracy Before vs After Sanitation")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

# Optional: Show which features remain after sanitation
print("\nFeatures after sanitation:")
print(X_imputed.columns)

#=====================================================
# Step 6: Export the sanitized and imputed dataset
#=====================================================
# Add back the Outcome column
final_df = X_imputed.copy()
final_df['Outcome'] = y

# Export to a new CSV file
final_df.to_csv("sanitized_data.csv", index=False)
print("\nSanitized dataset exported to 'sanitized_data.csv'")
