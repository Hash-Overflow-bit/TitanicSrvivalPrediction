import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_train=pd.read_csv(r"C:\Users\Hashir Mehboob\Desktop\TitanicSrvivalPrediction\train.csv")

# print(data_train)

# Impute numeric 'Age' before encoding
data_train["Age"] = data_train["Age"].fillna(data_train["Age"].median())

# Handle missing values for Cabin and Embarked
data_train["Cabin"] = data_train["Cabin"].fillna("Unknown")
data_train["Embarked"] = data_train["Embarked"].fillna(data_train["Embarked"].mode()[0])

# Encoding
ohe = OneHotEncoder(handle_unknown="ignore")
cat_cols = ["Name", "Sex", "Ticket", "Embarked", "Cabin"]
X_cat = data_train[cat_cols]  # Missing values already handled above
X_ohe = ohe.fit_transform(X_cat)

# Build a DataFrame from the encoded matrix
feature_names = ohe.get_feature_names_out(cat_cols)
X_ohe_df = pd.DataFrame(X_ohe.toarray(), columns=feature_names, index=data_train.index)

# Combine with the non-categorical columns
data_encoded = pd.concat([data_train.drop(columns=cat_cols), X_ohe_df], axis=1)

# Save encoded dataset (original CSV is not modified unless explicitly written)
data_encoded.to_csv("train_encoded.csv", index=False)


# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Survival rate by Sex
sns.barplot(x='Sex', y='Survived', data=data_train, ax=axes[0, 0])
axes[0, 0].set_title('Survival Rate by Sex')
axes[0, 0].set_ylabel('Survival Rate')

# 2. Age distribution by Survival
sns.histplot(data=data_train, x='Age', hue='Survived', kde=True, bins=30, ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Survival')

# 3. Survival rate by Passenger Class
sns.barplot(x='Pclass', y='Survived', data=data_train, ax=axes[1, 0])
axes[1, 0].set_title('Survival Rate by Passenger Class')
axes[1, 0].set_ylabel('Survival Rate')

# 4. Fare distribution by Survival
sns.boxplot(x='Survived', y='Fare', data=data_train, ax=axes[1, 1])
axes[1, 1].set_title('Fare Distribution by Survival')
axes[1, 1].set_ylim(0, 300)

plt.tight_layout()
plt.savefig('titanic_analysis.png', dpi=150)
print("Visualization saved as 'titanic_analysis.png'")
# plt.show()  # Commented to avoid blocking execution


# ===== MODEL TRAINING =====
print("\n" + "="*60)
print("MODEL TRAINING & EVALUATION")
print("="*60)

# Prepare features and target
X = data_encoded.drop(['Survived', 'PassengerId'], axis=1)
y = data_encoded['Survived']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Train Logistic Regression
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# Accuracy
lr_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

# Cross-validation score
lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std():.4f})")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature coefficients (Logistic Regression)
print("\n--- Top 10 Most Influential Features ---")
feature_coef = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False).head(10)

print(feature_coef.to_string(index=False))

print("\n" + "="*60)
print(f"MODEL: Logistic Regression with {lr_accuracy*100:.2f}% accuracy")
print("="*60)

# ===== TEST SET PREDICTIONS =====
print("\n" + "="*60)
print("MAKING PREDICTIONS ON TEST SET")
print("="*60)

# Load test data
data_test = pd.read_csv(r"C:\Users\Hashir Mehboob\Desktop\TitanicSrvivalPrediction\test.csv")
test_passenger_ids = data_test['PassengerId'].copy()

print(f"\nTest set size: {len(data_test)} samples")

# Apply same preprocessing as training data
data_test["Age"] = data_test["Age"].fillna(data_test["Age"].median())
data_test["Cabin"] = data_test["Cabin"].fillna("Unknown")
data_test["Embarked"] = data_test["Embarked"].fillna(data_test["Embarked"].mode()[0])
data_test["Fare"] = data_test["Fare"].fillna(data_test["Fare"].median())  # In case of missing Fare

# Apply OneHotEncoder (must use the same encoder fitted on training data)
X_cat_test = data_test[cat_cols]
X_ohe_test = ohe.transform(X_cat_test)

# Build DataFrame from encoded matrix
X_ohe_test_df = pd.DataFrame(X_ohe_test.toarray(), columns=feature_names, index=data_test.index)

# Combine with non-categorical columns
data_test_encoded = pd.concat([data_test.drop(columns=cat_cols), X_ohe_test_df], axis=1)

# Prepare features (remove PassengerId, no Survived column in test)
X_test_final = data_test_encoded.drop(['PassengerId'], axis=1)

# Ensure same column order as training
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

# Make predictions
predictions = lr_model.predict(X_test_final)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)
print("\nPredictions saved to 'submission.csv'")
print(f"Predicted {predictions.sum()} survivors out of {len(predictions)} passengers")
print(f"Survival rate: {predictions.mean()*100:.2f}%")

print("\n" + "="*60)
print("COMPLETED")
print("="*60)