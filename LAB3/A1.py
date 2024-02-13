
import pandas as pd
import numpy as np
 
file_path ="C:\Users\\akhil\\OneDrive\\Documents\ML\LAB\lab2\Lab Session1 Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Purchase data")
A = df.iloc[:, 1:4].values  
C = df.iloc[:, 4].values  
dimensionality = A.shape[1]
num_vectors = A.shape[0]
rank_A = np.linalg.matrix_rank(A)
pseudo_inverse_A = np.linalg.pinv(A)
cost_vector = np.dot(pseudo_inverse_A, C)
print("Dimensionality of the vector space:", dimensionality)
print("Number of vectors in the vector space:", num_vectors)
print("Rank of Matrix A:", rank_A)
print("Cost of each product:", cost_vector)
 