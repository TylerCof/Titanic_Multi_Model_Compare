from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

#Step 1: Data
# Load the processed data
processed_data_path = 'final_data.csv'  # making sure the right path is loaded
df = pd.read_csv(processed_data_path)

# Separate features and target variable
X = df.drop('Survived', axis=1).values
y = df['Survived'].values

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Define classifiers
lr = LogisticRegression(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
svc = SVC(probability=True, random_state=42)  # Enable probability for soft voting
rf = RandomForestClassifier(random_state=42)

# Step 3: Voting Classifier with Hard Voting
hard_voting_clf = VotingClassifier(estimators=[('lr', lr), ('dt', dt), ('svc', svc), ('rf', rf)], voting='hard')
hard_voting_clf.fit(X_train, y_train)
y_pred_hard = hard_voting_clf.predict(X_test)
print("Hard Voting Accuracy:", accuracy_score(y_test, y_pred_hard))

# Step 4: Voting Classifier with Soft Voting
soft_voting_clf = VotingClassifier(estimators=[('lr', lr), ('dt', dt), ('svc', svc), ('rf', rf)], voting='soft')
soft_voting_clf.fit(X_train, y_train)
y_pred_soft = soft_voting_clf.predict(X_test)
print("Soft Voting Accuracy:", accuracy_score(y_test, y_pred_soft))

# Step 5: Display Confidence Levels for Soft Voting
y_pred_proba_soft = soft_voting_clf.predict_proba(X_test)
print("\nConfidence levels for the first few predictions by the soft voting classifier:")
for i, proba in enumerate(y_pred_proba_soft[:5]):
    print(f"Sample {i+1}: Class 0 = {proba[0]*100:.2f}%, Class 1 = {proba[1]*100:.2f}%")

# Save the hard voting classifier
with open('hard_voting_classifier.pkl', 'wb') as file:
    pickle.dump(hard_voting_clf, file)

# Save the soft voting classifier
with open('soft_voting_classifier.pkl', 'wb') as file:
    pickle.dump(soft_voting_clf, file)










