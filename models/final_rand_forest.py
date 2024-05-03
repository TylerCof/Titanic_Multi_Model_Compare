import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier  #--
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
file_path = 'final_data.csv'

# Check if the CSV file exists
if os.path.exists(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    print("CSV file loaded successfully.")
else:
    print(f"Error: CSV file '{file_path}' does not exist.")
    sys.exit()

# Separate features and target
X = df.drop("Survived", axis=1).values
y = df["Survived"].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random Forest Classifier
#rf_clf = RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_jobs=-1)
rf_clf = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_jobs=-1)

# Training the classifier
rf_clf.fit(X_train, y_train)

# Predictions on the test set
y_pred = rf_clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Compute Precision-Recall values
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# Plot the Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall Curve (RandomForest Bagging Classifier)')
plt.legend()
plt.grid(True)
plt.show()


# Save the trained model to a file using pickle
model_file = 'forest_model.pkl'
with open(model_file, 'wb') as file:
    pickle.dump(rf_clf, file)

print(f"Model saved to {model_file}")
