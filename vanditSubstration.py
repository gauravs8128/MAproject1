def subtract_matrices(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return "Matrices must have the same dimensions for subtraction."
    
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[0])):
            row.append(matrix1[i][j] - matrix2[i][j])
        result.append(row)
    
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

result = subtract_matrices(matrix1, matrix2)
for row in result:
    print(row)
