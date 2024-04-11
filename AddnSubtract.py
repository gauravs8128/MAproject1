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

# Get the first matrix from the user
print("Enter the first matrix:")
matrix1 = get_matrix_from_user()
if matrix1 is not None:
    # Convert the list of lists to a NumPy array
    matrix1 = np.array(matrix1)

    # Get the second matrix from the user
    print("\nEnter the second matrix:")
    matrix2 = get_matrix_from_user()
    if matrix2 is not None:
        # Convert the list of lists to a NumPy array
        matrix2 = np.array(matrix2)

        # Check if the matrices have the same shape
        if matrix1.shape != matrix2.shape:
            print("Error: Matrices must have the same shape for addition and subtraction.")
        else:
            # Perform matrix addition
            matrix_sum = np.add(matrix1, matrix2)
            print("\nMatrix Addition:")
            print(matrix_sum)

            # Perform matrix subtraction
            matrix_difference = np.subtract(matrix1, matrix2)
            print("\nMatrix Subtraction:")
            print(matrix_difference)
