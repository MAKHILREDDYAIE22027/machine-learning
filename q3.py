import numpy as np
def matrix_power(A, m):
    n = len(A)
    if A.shape != (n, n):
        raise ValueError("Matrix A must be square")
    result = np.eye(n)
    while m > 0:
        if m % 2 == 1:
            result = np.dot(result, A)
        A = np.dot(A, A)
        m //= 2
    return result

A = np.array([[1, 2], [3, 4]])  # Example square matrix
m = 2 # Example power
result = matrix_power(A, m)
print("A^m:")
print(result)