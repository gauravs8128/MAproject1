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
        new_matrix = np.array(matrix)
    return new_matrix

print("Enter 1 for addition")
print("Enter 2 for subtraction")
print("Enter 3 for multiplication")
print("Enter 4 for inverse")
print("Enter 5 for determinant")
print("Enter 6 for rank of matrix")
print("Enter 7 for LU factorisation")
print("Enter 8 for eigen values and eigenvectors")
print("Enter 9 to Check whether a matrix is diagonalizable or not")


# Perform Addition
number = int(input("Enter operation: "))
if number == 1:
    matrix1 = get_matrix_from_user()
    matrix2 = get_matrix_from_user()
    if matrix1 is not None and matrix2 is not None:
        matrix1 = np.array(matrix1)
        matrix2 = np.array(matrix2)
        if matrix1.shape != matrix2.shape:
            print("Error: Matrices must have the same shape for addition and subtraction.")
        else:
            matrix_sum = matrix1 + matrix2 
            print("\nMatrix Addition:")
            print(matrix_sum)



elif number == 2:
    # Perform subtraction
    matrix1 = get_matrix_from_user()
    matrix2 = get_matrix_from_user()

    if matrix1.shape != matrix2.shape:
                print("Error: Matrices must have the same shape for addition and subtraction.")
    else:
                matrix_difference = matrix1 - matrix2
                print("\nMatrix Subtraction:")
                print(matrix_difference)

elif number == 3:
    # Perform multiplication
    matrix1 = get_matrix_from_user()
    matrix2 = get_matrix_from_user()

    # Check if inner dimensions are compatible for multiplication
    if len(matrix1[0]) != len(matrix2):
     print("Error: Matrices cannot be multiplied. Inner dimensions must match.")
     exit()

   # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]

    # Perform element-wise multiplication and summation for each cell of the result matrix
    for i in range(len(matrix1)):
     for j in range(len(matrix2[0])):
      for k in range(len(matrix2)):
       result[i][j] += matrix1[i][k] * matrix2[k][j]

    # Print the resulting product matrix
    print("Resultant product matrix:")
    for row in result:
     print(row)
  
  
elif number == 4:
    # Calculate inverse
    A = get_matrix_from_user()
    n = A.shape[0]

    # Check if the matrix is square
    if A.shape[0] != A.shape[1]:
        print("Error: The matrix must be square to find its inverse.")
        exit()
    else:
        
     augmented_matrix = np.concatenate((A, np.identity(n)), axis=1)
    
     for i in range(n):
        # Pivot row operations
        pivot_row = i
        for j in range(i + 1, n):
            if abs(augmented_matrix[j, i]) > abs(augmented_matrix[pivot_row, i]):
                pivot_row = j
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]
        
        # Divide row by the pivot element
        pivot_element = augmented_matrix[i, i]
        augmented_matrix[i] /= pivot_element
        
        # Subtract multiples of the pivot row from other rows to make the other entries in the column zero
        for j in range(n):
            if i != j:
                factor = augmented_matrix[j, i]
                augmented_matrix[j] -= factor * augmented_matrix[i]
    
    print(augmented_matrix[:, n:])

elif number == 5:
    # Calculate determinant
 matrix = get_matrix_from_user()
 def determinant(matrix):
    n = len(matrix) # len() function get length of the object.Here it represents, number of rows
    
    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0 #we cant use a uninitialised variable
    for j in range(n): # iterates from 0 to n-1
        # Calculate the cofactor matrix
        cofactor = [[matrix[i][k] for k in range(n) if k != j] for i in range(1, n)]
        # Calculate the determinant recursively
        det += ((-1) ** j) * matrix[0][j] * determinant(cofactor)
    
    return det
 deter = determinant(matrix)
 print("Determinant:", deter)


elif number == 6:
    # Calculate rank
    matrix1 = get_matrix_from_user()
    if matrix1 is not None:
        # Convert the list of lists to a NumPy array
        matrix = np.array(matrix1)

        # Calculate the rank of the matrix
        rank = np.linalg.matrix_rank(matrix1)
        print("\nRank of the matrix:")
        print(rank)


elif number == 7:
     # Perform LU factorization
     """
    Performs LU decomposition of a square matrix using partial pivoting.

    Args:
        matrix: A NumPy array representing the square matrix to decompose.

    Returns:
        A tuple containing three NumPy arrays:
            - L (lower triangular matrix)
            - U (upper triangular matrix)

    Raises:
        ValueError: If the input matrix is not square.
    """
     matrix1 = get_matrix_from_user()

    # Check if the matrix is square
     if matrix1.shape[0] != matrix1.shape[1]:
            print("Error: LU factorization requires a square matrix.")
     else:

      # Create copies to avoid modifying the original matrix

      n = matrix1.shape[0]
      L = np.identity(n) #identity matrix of dimension n*n
      U = np.copy(matrix1) #creates copy of matrix 1

    # Perform LU decomposition with partial pivoting
     for col in range(n - 1):
        # Find the pivot element (largest element in the current column, excluding diagonal)
        max_index = np.argmax(np.abs(U[col + 1:, col])) + col + 1  #col + 1 for zero based indexing
        if np.abs(U[col, col]) < 1e-10:  # Checks if the absolute value of the diagonal element is near zero or not.(near-zero pivot)
            # Near zero value can cause unstability
            raise ValueError("Matrix is singular (near-zero pivot encountered)")

        # Swap rows if necessary (pivoting)
        if max_index != col: # largest element should not on diagonal
            U[[col, max_index]] = U[[max_index, col]]
            L[[col, max_index]] = L[[max_index, col]]

        # Update lower triangular matrix (L)
        for row in range(col + 1, n):
            factor = U[row, col] / U[col, col] # Gets the factor through which elements below pivot can become zero
            L[row, col] = factor
            U[row, col:] -= factor * U[col, col:]  # Update remaining elements
        
        print("\nLU Factorization:")
        print("L (Lower triangular matrix):")
        print(L)
        print("U (Upper triangular matrix):")
        print(U)
       

elif number == 8:
    # Calculate eigenvalues and eigenvectors
    matrix1 = get_matrix_from_user()
    if matrix1 is not None: #It means that it is initialised
        # Convert the list of lists to a NumPy array
        matrix1 = np.array(matrix1)

        # Calculate eigenvalues and eigenvectors through numpy function
        eigenvalues, eigenvectors = np.linalg.eig(matrix1)
        print("\nEigenvalues:")
        print(eigenvalues)
        print("\nEigenvectors:")
        print(eigenvectors)

elif number == 9:
    #Check whether a matrix is diagonalizable

    '''
    A matrix is diagonalizable if and only if 
    for each eigenvalue the dimension of the eigenspace is equal to
    the multiplicity of the eigenvalue.

    or we can say that for every eigenvalue, there should exist a eigenvector
    '''
    A = get_matrix_from_user()
    def is_diagonalizable(A):
     """
  Checks if a square matrix is diagonalizable.

  Args:
      A (np.ndarray): The square matrix to be checked.

  Returns:
      bool: True if the matrix is diagonalizable, False otherwise.
  """
    # Check if matrix is square
    if A.shape[0] != A.shape[1]:
     print("Entered matrix is not a square matrix")
     exit()

    # Find eigenvalues and their multiplicities
    eigenvalues, eigenvectors = np.linalg.eig(A)
    unique_eigenvalues, counts = np.unique(eigenvalues, return_counts=True) # returns unique elements with corresponding counts

    # Check if any eigenvalue has multiplicity greater than its eigenspace dimension
    for i, eigenvalue in enumerate(unique_eigenvalues):
     eigenspace_dim = np.linalg.matrix_rank(eigenvectors[:, eigenvalues == eigenvalue]) #calculated dimension of the eigenspace associated with eigenvalue
     if counts[i] > eigenspace_dim:
      print("eigenvalue has multiplicity greater than its eigenspace dimension, its not diagonalizable")
      exit()

    # All conditions met, matrix is diagonalizable
    print("YES, The matrix is Diagonalizable")  
