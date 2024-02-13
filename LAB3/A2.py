import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
 
file_path = "C:\\Users\\navee\\OneDrive\\Desktop\\mlcode\\lab3\\Lab Session1 Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Purchase data")
 
A = df.iloc[:, 1:4].values  
C = df.iloc[:, 4].values  
 
pseudo_inverse_A = np.linalg.pinv(A)
model_vector_X = np.dot(pseudo_inverse_A, C)
 
print("Model Vector X:")
print(model_vector_X)
df['Customer_Class'] = np.where(df['Payment (Rs)'] > 200, 'RICH', 'POOR')
 
X = df.iloc[:, 1:4].values  
y = df['Customer_Class'].values  
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)
 
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
 
print("\nLogistic Regression Classifier Results:")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_str)