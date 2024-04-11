def matrix_multiplication(matrix1, matrix2):
    # Check if matrices can be multiplied
    if len(matrix1[0]) != len(matrix2):
        print("Matrices cannot be multiplied. Number of columns in the first matrix must be equal to the number of rows in the second matrix.")
        return None
    
    # Initialize result matrix with zeros
    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
    
    # Perform multiplication
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    
    return result

import numpy as np

R1 = int(input("Enter the number of rows of first matrix:"))
C1 = int(input("Enter the number of columns of first matrix:"))
 
 
print("Enter the entries in a single line (separated by space): ")
 
# User input of entries in a 
# single line separated by space
entries = list(map(int, input().split()))
 
# For printing the matrix
matrix1 = np.array(entries).reshape(R1, C1)
print(matrix1)

R2 = int(input("Enter the number of rows of second matrix:"))
C2 = int(input("Enter the number of columns of second matrix:"))
 
 
print("Enter the entries in a single line (separated by space): ")
 
# User input of entries in a 
# single line separated by space
entries = list(map(int, input().split()))
 
# For printing the matrix
matrix2 = np.array(entries).reshape(R2, C2)
print(matrix2)

result_matrix = matrix_multiplication(matrix1, matrix2)

# Print result
if result_matrix:
    print("Result of matrix multiplication:")
    for row in result_matrix:
        print(row)
