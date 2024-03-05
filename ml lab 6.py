# Define the inputs and outputs for the AND gate
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 0, 0, 1]

# Define the initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
alpha = 0.05

# Define the activation functions
import math

def bipolar_step(x):
  if x < 0:
    return -1
  else:
    return 1

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def relu(x):
  if x < 0:
    return 0
  else:
    return x

# Define the perceptron function
def perceptron(A, B, activation):
  # Compute the weighted sum
  weighted_sum = W0 + W1 * A + W2 * B
  # Apply the activation function
  output = activation(weighted_sum)
  return output

# Define a function to calculate the sum of squared errors
def error(activation):
  total_error = 0
  for i in range(len(inputs)):
    # Get the input and output pair
    A, B = inputs[i]
    T = outputs[i]
    # Get the perceptron output
    Z = perceptron(A, B, activation)
    # Calculate the squared error
    squared_error = (T - Z) ** 2
    # Add to the total error
    total_error += squared_error
  return total_error

# Define a function to train the perceptron with a given activation function
def train(activation):
  # Define global variables for the weights
  global W0, W1, W2
  # Define a list to store the errors for each epoch
  errors = []
  # Define a variable to track the number of epochs
  epochs = 0
  # Train the perceptron until the error is zero
  while error(activation) > 0:
    # Increment the number of epochs
    epochs += 1
    # Store the current error
    errors.append(error(activation))
    # Loop through the input and output pairs
    for i in range(len(inputs)):
      # Get the input and output pair
      A, B = inputs[i]
      T = outputs[i]
      # Get the perceptron output
      Z = perceptron(A, B, activation)
      # Update the weights
      if activation == bipolar_step:
        W0 = W0 + alpha * (T - Z) * 1
        W1 = W1 + alpha * (T - Z) * A
        W2 = W2 + alpha * (T - Z) * B
      elif activation == sigmoid:
        W0 = W0 + alpha * (T - Z) * Z * (1 - Z) * 1
        W1 = W1 + alpha * (T - Z) * Z * (1 - Z) * A
        W2 = W2 + alpha * (T - Z) * Z * (1 - Z) * B
      elif activation == relu:
        W0 = W0 + alpha * (T - Z) * 1
        W1 = W1 + alpha * (T - Z) * A
        W2 = W2 + alpha * (T - Z) * B
  # Return the final weights, number of epochs, and errors
  return W0, W1, W2, epochs, errors

# Define a list of activation functions
activations = [bipolar_step, sigmoid, relu]

# Loop through the activation functions
for activation in activations:
  # Reset the weights to the initial values
  W0 = 10
  W1 = 0.2
  W2 = -0.75
  # Train the perceptron with the activation function
  W0, W1, W2, epochs, errors = train(activation)
  # Print the results
  print("Activation function:", activation._name_)
  print("Final weights:")
  print("W0 =", W0)
  print("W1 =", W1)
  print("W2 =", W2)
  print("Number of epochs:", epochs)
  # Plot the errors against the epochs
  import matplotlib.pyplot as plt
  plt.plot(range(1, epochs + 1), errors)
  plt.xlabel("Epoch")
  plt.ylabel("Error")
  plt.title("Perceptron Learning Curve with " + activation._name_)
  plt.show()