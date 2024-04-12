import numpy as np

R = int(input("Enter the number of rows of first matrix:"))
C = int(input("Enter the number of columns of first matrix:"))
 
 
print("Enter the entries in a single line (separated by space): ")

entries = list(map(int, input().split()))

matrix = np.array(entries).reshape(R, C)

print(matrix)
 
def determinant(matrix):
    # Base case: if matrix is 2x2
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    # Recursive case: using cofactor expansion along the first row
    for col in range(len(matrix)):
        sign = (-1) ** col
        sub_matrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        det += sign * matrix[0][col] * determinant(sub_matrix)

    return det

print("Determinant of the matrix:")
print(determinant(matrix))


