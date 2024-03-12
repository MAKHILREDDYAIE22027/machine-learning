import numpy as np

def matrix_power(A, m):
    # Convert the matrix to a numpy array
    A_np = np.array(A)

    # Check if the matrix is square
    if A_np.shape[0] != A_np.shape[1]:
        raise ValueError("Input matrix must be square")

    # Calculate the matrix raised to the power of m
    result_matrix = np.linalg.matrix_power(A_np, m)

    return result_matrix

# Take user input for the square matrix
n = int(input("Enter the size of the square matrix (n): "))
print(f"Enter the elements of the {n}x{n} matrix:")
matrix_A = [[float(input(f"Enter element A[{i + 1}][{j + 1}]: ")) for j in range(n)] for i in range(n)]

# Take user input for the positive integer m
power_m = int(input("Enter a positive integer (m): "))

# Calculate A^m
result = matrix_power(matrix_A, power_m)

# Print the result
print(f"\nA^{power_m}:\n{result}")
