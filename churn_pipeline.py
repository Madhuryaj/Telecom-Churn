# churn_pipeline.py

# Author - By Madhurya Jagadeesh

# ----------------------------------------
# 1. Business Understanding
# ----------------------------------------
# Goal: Predict customer churn to help retain users.
# Dataset: churn.xlsx
# Deliverables: EDA visualizations, ML model, evaluation report

# ----------------------------------------
# 2. Data Understanding
# ----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
file_path = 'churn.xlsx'
df = pd.read_excel(file_path)

# Basic overview
print("\n Data Info:")
print(df.info())
print("\n Target Distribution:")
print(df['Churn'].value_counts())
print("\n Summary Statistics:")
print(df.describe())

# ----------------------------------------
# Visualizations: Churn Factors
# ----------------------------------------

# 1. Churn by Tariff Plan
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='TariffPlan', hue='Churn')
plt.title('Churn Rate by Tariff Plan')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Churn by Age Group
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='AgeGroup', hue='Churn')
plt.title('Churn Rate by Age Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Tenure vs Churn
if 'Tenure' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='Tenure')
    plt.title('Tenure vs Churn')
    plt.tight_layout()
    plt.show()

# 4. Churn by Complaints
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Complains', hue='Churn')
plt.title('Churn by Complaints')
plt.tight_layout()
plt.show()

# 5. Churn by Status
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Status', hue='Churn')
plt.title('Churn by Status')
plt.tight_layout()
plt.show()

# ----------------------------------------
# Extended Statistical Analysis
# ----------------------------------------

# 6. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# 7. Pairplot for key numerical features (if present)
selected_features = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
if all(f in df.columns for f in selected_features):
    sns.pairplot(df[selected_features], hue='Churn', diag_kind='kde', palette='husl')
    plt.suptitle('Pairplot of Key Features vs Churn', y=1.02)
    plt.tight_layout()
    plt.show()

# 8. Grouped summary statistics
print("\n Mean Feature Values by Churn Class:")
print(df.groupby('Churn').mean(numeric_only=True).T.sort_values(by=1, ascending=False).head(10))

# ----------------------------------------
# 3. Data Preparation
# ----------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# One-hot encoding
df = pd.get_dummies(df, columns=['Complains', 'TariffPlan', 'AgeGroup', 'Status'], drop_first=True)

# Drop missing values
df.dropna(inplace=True)

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# SMOTE for balancing
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------
# 4. Modeling
# ----------------------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train_res)

# ----------------------------------------
# Feature Importance
# ----------------------------------------
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
plt.title('Top 10 Important Features for Churn Prediction')
plt.tight_layout()
plt.show()

# ----------------------------------------
# 5. Evaluation
# ----------------------------------------
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test_scaled)
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))

# ----------------------------------------
# 6. Deployment
# ----------------------------------------
import joblib

model_path = os.path.join(os.getcwd(), 'churn_rf_model.pkl')
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n Model and Scaler saved to:\n• {model_path}\n• {scaler_path}")
