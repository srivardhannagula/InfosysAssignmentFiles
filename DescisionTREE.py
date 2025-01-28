import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Wine dataset
wine = load_wine()
X = wine.data  # Features
y = wine.target  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)

# Train the model
dt_clf.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt_clf.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Classifier Accuracy (Wine Dataset): {accuracy_dt:.2f}")

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_clf, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)
plt.title("Decision Tree Visualization (Wine Dataset)")
plt.show()
