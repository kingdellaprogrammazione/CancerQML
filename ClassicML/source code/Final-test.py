import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Load and initial preprocess
# ---------------------------
df = pd.read_csv('Breast_cancer.csv', sep=',')  # Adjust file name/path as needed
df.columns = df.columns.str.strip()  # Remove any trailing spaces

print("Columns:", df.columns.tolist())  # Check column names

# Identify the label column
label_col = 'Status'

# Separate features and target
X = df.drop(columns=[label_col])
y = df[label_col]

# Map target values from strings to numeric
y = y.map({'Dead': 0, 'Alive': 1})

# According to the dataset snippet you provided, the columns are:
# Age, Race, Marital St., T Stage, N Stage, 6th Stage, differenti,
# Grade, A Stage, Tumor Siz, Estrogen S, Progester, Regional N, Reginol N, Survival M, Status
#
# We'll treat the following as categorical (textual):
cat_cols = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage',
            'differentiate', 'A Stage', 'Estrogen Status', 'Progesterone Status',]
num_cols = ['Age', 'Grade', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']

# Ensure numeric columns are indeed numeric
for nc in num_cols:
    df[nc] = pd.to_numeric(df[nc], errors='coerce')

X_cat = X[cat_cols].astype(str)
X_num = X[num_cols]

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = encoder.fit_transform(X_cat)
cat_feature_names = encoder.get_feature_names_out(cat_cols)

X_encoded = pd.DataFrame(
    np.hstack([X_num.values, X_cat_encoded]),
    columns=num_cols + list(cat_feature_names)
)

# Force all columns in X_encoded to numeric, just in case
X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')

# ---------------------------
# Identify Feature Importances using XGBoost
# ---------------------------
X_train_fi, X_test_fi, y_train_fi, y_test_fi = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_fi, y_train_fi)

# Get feature importance
importances = xgb.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# ---------------------------
# Plot Feature Importances Before Elimination
# ---------------------------
top_n_before = 20 if len(feature_importance_df) > 20 else len(feature_importance_df)
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feature_importance_df.head(top_n_before), color="skyblue")
plt.title("Top 20 Feature Importances (Before Elimination)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ---------------------------
# Select top N features and eliminate the rest
# ---------------------------
top_n = 10
top_features = feature_importance_df.head(top_n)['feature'].tolist()

# Determine which features were eliminated
all_features_set = set(feature_importance_df['feature'])
top_features_set = set(top_features)
eliminated_features = all_features_set - top_features_set

print(f"\nRetained top {top_n} features:")
print(top_features)
print("\nEliminated features:")
print(list(eliminated_features))

# ---------------------------
# Plot the top N features after elimination
# ---------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feature_importance_df.head(top_n), color="skyblue")
plt.title("Top 10 Feature Importances (After Elimination)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ---------------------------
# Prepare for Sanitization
# ---------------------------
X_reduced = X_encoded[top_features]

# Replace zeros with NaN in numeric top features
numeric_top_features = [f for f in top_features if f in num_cols]
X_reduced_sanitized = X_reduced.copy()
for col in numeric_top_features:
    X_reduced_sanitized[col] = X_reduced_sanitized[col].replace(0, np.nan)

# Apply KNN Imputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_reduced_sanitized)
X_imputed = pd.DataFrame(X_imputed, columns=top_features)

# ---------------------------
# Train NN on unsanitized vs sanitized
# ---------------------------
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

def build_nn_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train NN on unsanitized data
model_u = build_nn_model(X_train_u.shape[1])
model_u.fit(X_train_u, y_train_u, epochs=10, batch_size=16, verbose=0)
y_pred_u = (model_u.predict(X_test_u) > 0.5).astype(int)
acc_u = accuracy_score(y_test_u, y_pred_u)

# Train NN on sanitized data
model_s = build_nn_model(X_train_s.shape[1])
model_s.fit(X_train_s, y_train_s, epochs=10, batch_size=16, verbose=0)
y_pred_s = (model_s.predict(X_test_s) > 0.5).astype(int)
acc_s = accuracy_score(y_test_s, y_pred_s)

print("\nAccuracy on unsanitized data:", acc_u)
print("Accuracy on sanitized data:", acc_s)

# ---------------------------
# Export the sanitized CSV
# ---------------------------
df_sanitized = pd.DataFrame(X_imputed, columns=top_features)
df_sanitized[label_col] = y.values
df_sanitized.to_csv('data_sanitized.csv', index=False)
