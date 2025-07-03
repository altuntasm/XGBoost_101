import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# Load the breast cancer dataset
data = load_breast_cancer()

# Convert to pandas DataFrame for easier handling
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0 = malignant, 1 = benign

# Split the dataset into training and testing sets (80% train, 20% test), stratify enables class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
) 

# Initialize the XGBoost classifier
model = XGBClassifier(
    use_label_encoder=False,  # Avoid warning
    eval_metric='logloss',    # Recommended for binary classification
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Fit the model to the training data
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# More detailed classification report
print(classification_report(y_test, y_pred, target_names=data.target_names))


# Plot feature importances
importances = model.feature_importances_
features = X.columns

# Sort and plot
sorted_idx = importances.argsort()[::-1]
plt.figure(figsize=(10, 6))
plt.barh(features[sorted_idx][:10], importances[sorted_idx][:10])
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()
plt.show()
