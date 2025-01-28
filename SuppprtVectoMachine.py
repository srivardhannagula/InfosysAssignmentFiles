import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset from a CSV file
# Ensure the file contains columns: sepal_length, sepal_width, petal_length, petal_width, species
iris_df = pd.read_csv("iris.csv")

# Extract features and labels
X = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values

y = iris_df["species"].values  # Labels

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Check for and handle missing values
X = pd.DataFrame(X).fillna(method='ffill').to_numpy()

# Use only the first two features (sepal_length and sepal_width) for visualization
X_visual = X[:, :2]  # Select only the first two features for a 2D plot

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_visual_train, X_visual_test = train_test_split(X_visual, test_size=0.3, random_state=42, stratify=y)

# Standardize the feature data (SVM works better with scaled data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_visual_train = scaler.fit_transform(X_visual_train)
X_visual_test = scaler.transform(X_visual_test)

# Initialize and train the SVM classifier on full feature set
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions on the full feature set
y_pred = svm_classifier.predict(X_test)

# Evaluate the classifier on full feature set
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print the results for full feature set
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Train a new SVM classifier on 2D feature set for visualization
svm_visual_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_visual_classifier.fit(X_visual_train, y_train)

# Visualize the decision boundary
def plot_decision_boundary(X, y, classifier, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.show()

plot_decision_boundary(X_visual_train, y_train, svm_visual_classifier, "SVM Decision Boundary (Training Set)")
