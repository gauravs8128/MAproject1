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

# Get matrix from the user
matrix = get_matrix_from_user()
if matrix is not None:
    # Convert the list of lists to a NumPy array
    matrix = np.array(matrix)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print("\nEigenvalues:")
    print(eigenvalues)
    print("\nEigenvectors:")
    print(eigenvectors)
