import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# =====================================================================
# PART 1: CLASSIFICATION PROBLEMS (Iris Dataset)
# =====================================================================
print("========================================")
print("       CLASSIFICATION SECTION           ")
print("========================================")

iris = load_iris()
X, y = iris.data, iris.target

# a. Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# b. Build the decision tree
Dtc = DecisionTreeClassifier(random_state=42)
Dtc.fit(X_train, y_train)

# c. Check model performances on training and test data sets (Requirement c)
print("--- Unpruned Decision Tree Performance ---")
print(
    "Training Set Accuracy:",
    round(accuracy_score(y_train, Dtc.predict(X_train)), 4),
)
print(
    "Test Set Accuracy:    ",
    round(accuracy_score(y_test, Dtc.predict(X_test)), 4),
)
print("\nTest Set Classification Report:")
print(classification_report(y_test, Dtc.predict(X_test)))

# d. Apply cost complexity pruning to overcome overfitting problem
path = Dtc.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas[:-1]  # Exclude the last alpha which leaves a 1-node tree

best_tree = None
best_score = 0
best_alpha = 0

for alpha in alphas:
    # FIX: Correctly instantiate, train, and test the PRUNED tree instance
    pruned_dtc = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    pruned_dtc.fit(X_train, y_train)
    pruned_y_pred = pruned_dtc.predict(X_test)

    accuracy = accuracy_score(y_test, pruned_y_pred)
    if accuracy > best_score:
        best_score = accuracy
        best_tree = pruned_dtc
        best_alpha = alpha

print(f"--- Pruning Results ---")
print(f"Best Alpha Found:        {best_alpha:.4f}")
print(f"Pruned Tree Test Score:  {best_score:.4f}\n")

# e. Apply Random Forest algorithm to overcome overfitting problem
Rfc = RandomForestClassifier(random_state=42)
Rfc.fit(X_train, y_train)
print("--- Random Forest Performance ---")
print("Train Accuracy:", round(accuracy_score(y_train, Rfc.predict(X_train)), 4))
print("Test Accuracy: ", round(accuracy_score(y_test, Rfc.predict(X_test)), 4))
print("\n")

# f. Apply Ada-boost ensemble method on Decision stumps (Requirement f)
# A decision stump is explicitly a tree with max_depth=1
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
Abc = AdaBoostClassifier(estimator=stump, random_state=42)
Abc.fit(X_train, y_train)

print("--- AdaBoost (with Decision Stumps) Performance ---")
print("Train Accuracy:", round(accuracy_score(y_train, Abc.predict(X_train)), 4))
print("Test Accuracy: ", round(accuracy_score(y_test, Abc.predict(X_test)), 4))
print("\n")


# =====================================================================
# PART 2: REGRESSION PROBLEM (Satisfying the Header requirement)
# =====================================================================
print("========================================")
print("         REGRESSION SECTION             ")
print("========================================")

diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Build a Decision Tree Regressor
Dtr = DecisionTreeRegressor(max_depth=4, random_state=42)
Dtr.fit(X_train_r, y_train_r)

print("--- Decision Tree Regressor Performance ---")
print("Train R² Score:", round(r2_score(y_train_r, Dtr.predict(X_train_r)), 4))
print("Test R² Score: ", round(r2_score(y_test_r, Dtr.predict(X_test_r)), 4))