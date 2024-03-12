from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris  # Example dataset, you can replace it with your dataset

# Load the dataset (example with Iris dataset)
filepath=r"C:/Users/Aaron/Downloads/DATA/DATA/code_comm.csv"
data=pd.read_csv(filepath)
feature_name = input("Enter the name of the feature you want to analyze: ")
target_class_name = input("Enter the name of the target class column: ")



feature_data = data[feature_name]
target_data=data[target_class_name]
X=feature_data
y=target_data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model (you can replace this with any other classifier)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute confusion matrix
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

# Compute precision, recall, and F1-score
train_precision = precision_score(y_train, y_train_pred, average='macro')
test_precision = precision_score(y_test, y_test_pred, average='macro')

train_recall = recall_score(y_train, y_train_pred, average='macro')
test_recall = recall_score(y_test, y_test_pred, average='macro')

train_f1_score = f1_score(y_train, y_train_pred, average='macro')
test_f1_score = f1_score(y_test, y_test_pred, average='macro')

# Print results
print("Training Confusion Matrix:")
print(train_conf_matrix)
print("Training Precision:", train_precision)
print("Training Recall:", train_recall)
print("Training F1-Score:", train_f1_score)
print("\nTest Confusion Matrix:")
print(test_conf_matrix)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1-Score:", test_f1_score)