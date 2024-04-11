//Code One.
import numpy as np
import scipy.linalg as la

def get_matrix_from_user():
    print("Enter the elements of the matrix row-wise (separated by spaces):")
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    print("Enter the elements:")
    matrix = []
    for i in range(rows):
        row = list(map(float, input().split()))
        if len(row) != cols:
            print("Error: Number of elements in each row should be equal to the number of columns.")
            return None
        matrix.append(row)
    return matrix

# Get matrix from the user
matrix = get_matrix_from_user()
if matrix is not None:
    # Convert the list of lists to a NumPy array
    matrix = np.array(matrix)

    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print("Error: LU factorization requires a square matrix.")
    else:
        # Perform LU factorization
        try:
            P, L, U = la.lu(matrix)
            print("\nLU Factorization:")
            print("P (Permutation matrix):")
            print(P)
            print("L (Lower triangular matrix):")
            print(L)
            print("U (Upper triangular matrix):")
            print(U)
        except np.linalg.LinAlgError:
            print("Error: LU factorization failed. The matrix may be singular.")


//Code Two 

import numpy as np

def lu_factorization(matrix):
    n = len(matrix)
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        U[i, i:] = matrix[i, i:]
        L[i+1:, i] = matrix[i+1:, i] / U[i, i]
        for j in range(i+1, n):
            matrix[j, i:] -= L[j, i] * U[i, i:]
    return L, U

def get_matrix_from_user():
    print("Enter the elements of the matrix row-wise (separated by spaces):")
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    print("Enter the elements:")
    matrix = []
    for i in range(rows):
        row = list(map(float, input().split()))
        if len(row) != cols:
            print("Error: Number of elements in each row should be equal to the number of columns.")
            return None
        matrix.append(row)
    return matrix

# Get matrix from the user
matrix = get_matrix_from_user()
if matrix is not None:
    # Convert the list of lists to a NumPy array
    matrix = np.array(matrix)

    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print("Error: LU factorization requires a square matrix.")
    else:
        # Perform LU factorization
        try:
            L, U = lu_factorization(matrix)
            print("\nLU Factorization:")
            print("L (Lower triangular matrix):")
            print(L)
            print("U (Upper triangular matrix):")
            print(U)
        except np.linalg.LinAlgError:
            print("Error: LU factorization failed. The matrix may be singular.")
