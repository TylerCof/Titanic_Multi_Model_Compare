import os
import sys
import pandas as pd
from missingDataScript import find_missing_values
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Load the processed data
processed_data_path = 'final_data.csv'  #making sure the right path is loaded
df = pd.read_csv(processed_data_path)

# Separate features and target variable
X = df.drop('Survived', axis=1).values
y = df['Survived'].values

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a pipeline that includes scaling and the SVM model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

# Define the parameter grid for C and gamma

param_grid = {
    'svc__C': [.1,1,5,10,100],
    'svc__gamma': ['scale', 'auto', .1, .01],
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
    'svc__degree': [1,2,3,4,5]
    }

'''

param_grid = {
    'svc__C': [0.001, 0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
    }

param_grid = {
    'svc__C': [0.01, 0.1, 1, 10],
    'svc__kernel': ['linear'],
    'svc__gamma': ['scale', 'auto', 0.1]
}
'''
# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Predict on the testing set using the best found parameters
y_pred = grid_search.predict(X_test)

# Evaluate the model
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


