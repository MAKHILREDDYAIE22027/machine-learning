import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import accuracy_score

class Perceptron(BaseEstimator):
    def __init__(self, num_features, learning_rate=0.01, epochs=1000):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(weighted_sum)

    def train(self, inputs, labels):
        self.weights = np.random.rand(self.num_features)
        self.bias = np.random.rand()
        for _ in range(self.epochs):
            for x, y in zip(inputs, labels):
                prediction = self.predict(x)
                error = y - prediction
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error

    def fit(self, X, y):
        self.train(X, y)
        return self

    def predict(self, X):
        return np.where(self.sigmoid(np.dot(X, self.weights) + self.bias) >= 0.5, 1, 0)

    def get_params(self, deep=True):
        return {
            'num_features': self.num_features,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs
        }

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

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'learning_rate': uniform(0.001, 0.1),
    'epochs': [100, 500, 1000, 2000]
}

# Create an instance of the Perceptron class
perceptron = Perceptron(num_features=X_train.shape[1])

# Create RandomizedSearchCV instance with a dummy scoring function
random_search = RandomizedSearchCV(estimator=perceptron, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy')

# Fit RandomizedSearchCV to training data
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Initialize perceptron with best parameters
best_perceptron = Perceptron(num_features=X_train.shape[1], **best_params)


# Train the perceptron with best parameters
best_perceptron.fit(X_train, y_train)

# Test the perceptron with best parameters on test data
y_pred = best_perceptron.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of best perceptron on test data: {accuracy}")
