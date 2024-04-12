import numpy as np

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

print("Enter 1 for addition")
print("Enter 2 for subtraction")
print("Enter 3 for multiplication")
print("Enter 4 for division")
print("Enter 5 for inverse")
print("Enter 6 for determinant")
print("Enter 7 for rank of matrix")
print("Enter 8 for LU factorisation")
print("Enter 9 for eigen values and eigenvectors")

number = int(input("Enter a number: "))

if number == 1:
    matrix1 = get_matrix_from_user()
    matrix2 = get_matrix_from_user()


def add_matrices(matrix1, matrix2):
    # Check if the dimensions of both matrices are the same
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        print("Error: Matrices must have the same dimensions for addition.")
        return None

    # Initialize a result matrix with zeros
    result = [[0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))]

    # Iterate through each element of the matrices and add them
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result[i][j] = matrix1[i][j] + matrix2[i][j]

    return result
# Perform matrix addition
result = add_matrices(matrix1, matrix2)

# Print the result
if result:
    print("Matrix Addition Result:")
    for row in result:
        print(row)
elif number == 2:
    # Perform subtraction
    matrix1 = get_matrix_from_user()
    if matrix1 is not None:
        matrix1 = np.array(matrix1)
        matrix2 = get_matrix_from_user()
        if matrix2 is not None:
            matrix2 = np.array(matrix2)
            if matrix1.shape != matrix2.shape:
                print("Error: Matrices must have the same shape for addition and subtraction.")
            else:
                matrix_difference = np.subtract(matrix1, matrix2)
                print("\nMatrix Subtraction:")
                print(matrix_difference)

elif number == 3:
    # Perform multiplication
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

    R1 = int(input("Enter the number of rows of first matrix:"))
    C1 = int(input("Enter the number of columns of first matrix:"))

    print("Enter the entries in a single line (separated by space): ")

    entries = list(map(int, input().split()))

    matrix1 = np.array(entries).reshape(R1, C1)
    print(matrix1)

    R2 = int(input("Enter the number of rows of second matrix:"))
    C2 = int(input("Enter the number of columns of second matrix:"))

    print("Enter the entries in a single line (separated by space): ")

    entries = list(map(int, input().split()))

    matrix2 = np.array(entries).reshape(R2, C2)
    print(matrix2)

    result_matrix = matrix_multiplication(matrix1, matrix2)

    # Print result
    if result_matrix:
        print("Result of matrix multiplication:")
        for row in result_matrix:
            print(row)

elif number == 4:
    # Perform division
    pass

elif number == 5:
    # Calculate inverse
    matrix1 = get_matrix_from_user()
    matrix1 = np.array(matrix1)

    # Convert the list of lists to a NumPy array
    matrix1 = np.array(matrix1)

    # Check if the matrix is square
    if matrix1.shape[0] != matrix1.shape[1]:
        print("Error: The matrix must be square to find its inverse.")
    else:
        # Calculate the inverse of the matrix
        try:
            inverse_matrix = np.linalg.inv(matrix1)
            print("\nInverse of the matrix:")
            print(inverse_matrix)
        except np.linalg.LinAlgError:
            print("Error: The matrix is singular and does not have an inverse.")

elif number == 6:
    # Calculate determinant
    def det_recursive(A):
        """
        This function calculates the determinant of a matrix using recursion and block partitioned matrices.

        Args:
            A: A square numpy array representing the matrix.

        Returns:
            The determinant of the matrix.
        """
        n = len(A)
        if n == 1:
            return A[0, 0]
        elif n == 2:
            return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        else:
            # Split the matrix into submatrices
            A11 = A[:n//2, :n//2]
            A12 = A[:n//2, n//2:]
            A21 = A[n//2:, :n//2]
            A22 = A[n//2:, n//2:]

            # Calculate determinants of submatrices recursively
            det_A11 = det_recursive(A11)
            det_A22 = det_recursive(A22)

            # Calculate the determinant using the block matrix formula
            return det_A11 * det_A22 - np.linalg.det(A12 @ A21)

    # Create a sample 10x10 matrix
    A = np.random.rand(10, 10)

    # Calculate the determinant using the recursive function
    determinant = det_recursive(A)

    print("Determinant of the matrix:", determinant)
elif number == 7:
    # Calculate rank
    matrix1 = get_matrix_from_user()
    if matrix1 is not None:
        # Convert the list of lists to a NumPy array
        matrix = np.array(matrix1)

        # Calculate the rank of the matrix
        rank = np.linalg.matrix_rank(matrix1)
        print("\nRank of the matrix:")
        print(rank)

elif number == 8:
    # Perform LU factorization
    def lu_factorization(matrix1):
        n = len(matrix1)
        L = np.eye(n)
        U = np.zeros((n, n))

        for i in range(n):
            U[i, i:] = matrix1[i, i:]
            L[i+1:, i] = matrix1[i+1:, i] / U[i, i]
            for j in range(i+1, n):
                matrix1[j, i:] -= L[j, i] * U[i, i:]

        return L, U

    # Get matrix from the user
    matrix1 = get_matrix_from_user()
    if matrix1 is not None:
        # Convert the list of lists to a NumPy array
        matrix1 = np.array(matrix1)

        # Check if the matrix is square
        if matrix1.shape[0] != matrix1.shape[1]:
            print("Error: LU factorization requires a square matrix.")
        else:
            # Perform LU factorization
            try:
                L, U = lu_factorization(matrix1)
                print("\nLU Factorization:")
                print("L (Lower triangular matrix):")
                print(L)
                print("U (Upper triangular matrix):")
                print(U)
            except np.linalg.LinAlgError:
                print("Error: LU factorization failed. The matrix may be singular.")

elif number == 9:
    # Calculate eigenvalues and eigenvectors
    matrix1 = get_matrix_from_user()
    if matrix1 is not None:
        # Convert the list of lists to a NumPy array
        matrix1 = np.array(matrix1)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix1)
        print("\nEigenvalues:")
        print(eigenvalues)
        print("\nEigenvectors:")
        print(eigenvectors)
