import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

#=====================================================
# Step 1: Load the dataset
#=====================================================
df = pd.read_csv("ex.csv")

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

#=====================================================
# Step 2: Analyze zeros in the dataset
#=====================================================
# Count zeros in each column
zero_counts = (X == 0).sum()
print("Count of Zeroes in Each Feature:")
print(zero_counts)

# Plot zero counts
zero_counts.plot(kind='bar', title='Count of Zeroes in Each Feature')
plt.xlabel("Features")
plt.ylabel("Count of Zeros")
plt.show()

# We can also check the percentage of zeros:
zero_percentage = (X == 0).mean() * 100
print("\nPercentage of Zeroes in Each Feature:")
print(zero_percentage)

# If columns like BMI, BloodPressure, SkinThickness, Insulin have 0 values, these likely represent missing
# We can examine how these zeros distribute with respect to Outcome
for col in X.columns:
    fig, ax = plt.subplots()
    X_zero_flag = (X[col] == 0)
    # Compare outcome distribution when this feature is zero vs non-zero
    zero_outcome_counts = y[X_zero_flag].value_counts(normalize=True)
    nonzero_outcome_counts = y[~X_zero_flag].value_counts(normalize=True)

    # Create a small bar plot
    index = ['Outcome=0','Outcome=1']
    width = 0.35
    zero_vals = [zero_outcome_counts.get(0,0), zero_outcome_counts.get(1,0)]
    nonzero_vals = [nonzero_outcome_counts.get(0,0), nonzero_outcome_counts.get(1,0)]

    bar1 = ax.bar(np.arange(len(index)) - width/2, zero_vals, width, label='Zeros in ' + col)
    bar2 = ax.bar(np.arange(len(index)) + width/2, nonzero_vals, width, label='Non-Zeros in ' + col)

    ax.set_ylabel('Proportion')
    ax.set_title(f'Outcome Distribution with/without zeros in {col}')
    ax.set_xticks(np.arange(len(index)))
    ax.set_xticklabels(index)
    ax.legend()
    plt.show()

#=====================================================
# Step 3: Impute zeros to handle missingness
#=====================================================
# Identify columns to impute (e.g., certain biological measurements should never be zero)
# In many diabetes datasets, these columns are typically imputed:
columns_to_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Replace zeros with NaN in these columns
X_imputed = X.copy()
X_imputed[columns_to_impute] = X_imputed[columns_to_impute].replace(0, np.nan)

# Impute using median as a simple strategy
imputer = SimpleImputer(strategy='median')
X_imputed[columns_to_impute] = imputer.fit_transform(X_imputed[columns_to_impute])

#=====================================================
# Step 4: Compare model performance before and after imputation
#=====================================================

def train_and_evaluate(X_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return model

print("Performance Before Imputation:")
train_and_evaluate(X, y)

print("\nPerformance After Imputation:")
train_and_evaluate(X_imputed, y)

#=====================================================
# Step 5: Analyze if imputation improved feature distributions
#=====================================================
# For example, we can look at summary stats before and after imputation
print("\nSummary before imputation:")
print(X.describe())

print("\nSummary after imputation:")
print(X_imputed.describe())

# Optional: you could also visualize distributions before and after
fig, axs = plt.subplots(nrows=2, ncols=len(columns_to_impute), figsize=(15, 8))
for i, col in enumerate(columns_to_impute):
    # Before imputation distribution
    axs[0, i].hist(X[col], bins=20, color='blue', alpha=0.7)
    axs[0, i].set_title(f'{col} Before Imputation')

    # After imputation distribution
    axs[1, i].hist(X_imputed[col], bins=20, color='green', alpha=0.7)
    axs[1, i].set_title(f'{col} After Imputation')

plt.tight_layout()
plt.show()
