import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classifier(classifier, X, y):
    accuracy = cross_val_score(classifier, X, y, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(classifier, X, y, cv=5, scoring='precision').mean()
    recall = cross_val_score(classifier, X, y, cv=5, scoring='recall').mean()
    f1 = cross_val_score(classifier, X, y, cv=5, scoring='f1').mean()
    return accuracy, precision, recall, f1

# Load data from Excel sheet into DataFrame
df = pd.read_excel('customerdata.xlsx')

# Encoding labels to numeric values
label_encoding = {'Yes': 1, 'No': 0}
df['High Value Tx'] = df['High Value Tx'].map(label_encoding)

# Extracting features and labels
inputs = df.drop(columns=['Customer', 'High Value Tx']).values.astype(float)
labels = df['High Value Tx'].values

# Normalize inputs
inputs = inputs / inputs.max(axis=0)

# Define classifiers
classifiers = {
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Naïve Bayes": GaussianNB(),
    "CatBoost": CatBoostClassifier(logging_level='Silent'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Evaluate each classifier
results = {}
for clf_name, clf in classifiers.items():
    accuracy, precision, recall, f1 = evaluate_classifier(clf, inputs, labels)
    results[clf_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Display results in a tabular format
results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df)from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier

