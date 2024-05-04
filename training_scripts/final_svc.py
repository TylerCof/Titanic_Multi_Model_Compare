import os
import sys
import pandas as pd
from missingDataScript import find_missing_values
import joblib
import pickle
from pprint import pprint
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Load the dataset
processed_data_path = 'final_data.csv'

# Load the processed data
processed_data_path = 'final_data.csv'  #making sure the right path is loaded
df = pd.read_csv(processed_data_path)

print("Targeting selected features\n")
# Separate features and target variable
X = df.drop('Survived', axis=1).values
y = df['Survived'].values

print("Starting\n")
# Separate features and target variable
# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#--------Single

# Define the best hyperparameters from your grid search
best_params = {'svc__C': 10, 'svc__gamma': 0.1}

# Define the SVM model with the best hyperparameters
svm_model = SVC(C=best_params['svc__C'], gamma=best_params['svc__gamma'])

# Define the pipeline with scaling and the SVM model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', svm_model)
])

# Train the model on the entire training set
pipeline.fit(X_train, y_train)

# Predict on the testing set
y_pred = pipeline.predict(X_test)

#--------Single

# Evaluate the model
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#--------Single---Extended

# Pickle the model to a file
with open('final_svm.pkl', 'wb') as file:
    pickle.dump(pipeline, file)


