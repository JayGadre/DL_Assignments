import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# --- Summary Metrics ---
print("==================================================")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("==================================================")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# =====================================================================
# Crucial Addition: Print both correct and wrong predictions
# =====================================================================
print("==================================================")
print("DETAILED PREDICTIONS BREAKDOWN")
print("==================================================")

# Create a temporary DataFrame of the test set for clear printing
test_results = pd.DataFrame(X_test, columns=iris.feature_names)
test_results['Actual_Class'] = [target_names[i] for i in y_test]
test_results['Predicted_Class'] = [target_names[i] for i in y_pred]

# Separate Correct and Wrong predictions
correct_df = test_results[y_test == y_pred]
wrong_df = test_results[y_test != y_pred]

print(f"\n✔ CORRECT PREDICTIONS ({len(correct_df)} samples):")
if not correct_df.empty:
    print(correct_df.to_string(index=False))
else:
    print("None")

print(f"\n✖ WRONG PREDICTIONS ({len(wrong_df)} samples):")
print(wrong_df.to_string(index=False))
