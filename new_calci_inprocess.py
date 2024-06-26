import numpy as np

class MatrixCalculator: # A class to perform basic matrix operations.
  

  def __init__(self, matrix1, matrix2=None):
    """
    Initializes the calculator with two matrices.

    Args:
      matrix1: A NumPy array representing the first matrix.
      matrix2 (optional): A NumPy array representing the second matrix 
        (default: None).
    """
    self.matrix1 = matrix1
    self.matrix2 = matrix2
   
  

  def add(self):
    """
    Adds the two matrices.

    Returns:
      A NumPy array representing the sum of the matrices, 
      or None if the matrices cannot be added.
    """
    if self.matrix2 is None:
      return None
    if self.matrix1.shape != self.matrix2.shape:
      raise Exception("Matrices must have the same dimensions for addition.")
      return None
    return np.add(self.matrix1, self.matrix2)

  def subtract(self):
    """
    Subtracts the second matrix from the first matrix.

    Returns:
      A NumPy array representing the difference of the matrices, 
      or None if the matrices cannot be subtracted.
    """
    if self.matrix2 is None:
      return None
    if self.matrix1.shape != self.matrix2.shape:
      print("Matrices must have the same dimensions for subtraction.")
      return None
    return np.subtract(self.matrix1, self.matrix2)

  def multiply(self, matrix1=None, matrix2= None):
    """
    Multiplies the two matrices.

    Returns:
      A NumPy array representing the product of the matrices, 
      or None if the matrices cannot be multiplied.
    """
    if self.matrix2 is None:
      return None
    if self.matrix1.shape[1] != self.matrix2.shape[0]:  # Checking whether the matrix multiplication can be executed or not
      print("Inner dimensions must be equal for matrix multiplication.")
      return None
    return np.matmul(self.matrix1, self.matrix2)
  

  def division(self):
   """
   Divides matrix A by matrix B using the inverse of matrix B.

   Args:
    matrix_A: A NumPy array representing the first matrix (dividend).
    matrix_B: A NumPy array representing the second matrix (divisor).

   Returns:
    A NumPy array representing the result of the division, 
    or None if the division is not possible.
  """
   if matrix2 is not None:
    determinant_matrix2 = self.determinant()
   if determinant_matrix2 !=0 :
     inverse_B = np.linalg.inv(matrix2)
     
     return np.matmul(self.matrix1, inverse_B)
   else:
     raise Exception("matrixB is not invertible, hence division is not possible")



  def lu_decomposition(self):
    """
    Performs LU decomposition of a square matrix using partial pivoting.

    Args:
        matrix: A NumPy array representing the square matrix to decompose.

    Returns:
        A tuple containing three NumPy arrays:
            - L (lower triangular matrix)
            - U (upper triangular matrix)
            - P (permutation matrix for pivoting)

    Raises:
        ValueError: If the input matrix is not square.
    """

    if len(self.matrix1.shape) != 2 or self.matrix1.shape[0] != self.matrix1.shape[1]:
        raise ValueError("LU decomposition requires a square matrix.")

    # Create copies to avoid modifying the original matrix
    n = self.matrix1.shape[0]
    L = np.identity(n)
    U = np.copy(self.matrix1)

    # Perform LU decomposition with partial pivoting
    for col in range(n - 1):
        # Find the pivot element (largest element in the current column, excluding diagonal)
        max_index = np.argmax(np.abs(U[col + 1:, col])) + col + 1
        if np.abs(U[col, col]) < 1e-10:  # Check for degeneracy (near-zero pivot)
            raise ValueError("Matrix is singular (near-zero pivot encountered)")

        # Swap rows if necessary (pivoting)
        if max_index != col:
            U[[col, max_index]] = U[[max_index, col]]
            L[[col, max_index]] = L[[max_index, col]]

        # Update lower triangular matrix (L)
        for row in range(col + 1, n):
            factor = U[row, col] / U[col, col]
            L[row, col] = factor
            U[row, col:] -= factor * U[col, col:]  # Update remaining elements

    # Return the decomposed matrices
        return [L, U]


  def LeastSquareSolution(self):
    # we are provided with Ax = b having no soluti0n, 
    #b provided is not in the column space of A
    '''
    we want to find out a x' such that Ax' will get close to b , 
    meaning we want to find out a vector b' which is the closest to b in column space of A or which has a solution
    matrix
    
    A - matrix1
    X - matrix2
    B- input_matrix
    A^T * A*X' = A^T * b
    '''
    b = input_matrix()
  
    
    temp = np.transpose(self.matrix1) * self.matrix1
    
    if temp.shape[0] == b.shape[1]:
     temp2 =np.transpose(self.matrix1) * b
    else:
       return("Entered b doesnt have necessary shape")

    x = np.linalg.solve(temp, temp2)

    new_b = np.matmul(matrix1, x)
    return new_b


  def determinant(self):
     """
     Calculates the determinant of the first matrix.

     Returns:
      The determinant of the matrix, or None if the matrix is not square.
     """
     if self.matrix1.ndim != 2 or self.matrix1.shape[0] != self.matrix1.shape[1]:
      
       print("Matrix must be square to calculate determinant.")
       return None
     return np.linalg.det(self.matrix1)
  
  def linear_transformation(self):
    """
    Performs linear transformation (matrix multiplication) on matrix1 and matrix2.

    Returns:
        numpy.ndarray: The resulting matrix after the transformation.
        None: If the matrices have incompatible dimensions for multiplication.
    """

    if len(self.matrix1.shape) != 2 or len(self.matrix2.shape) != 2:
      raise ValueError("Error: Invalid dimensions for matrices.")

    if self.matrix1.shape[1] != self.matrix2.shape[0]:
      raise ValueError("Error: Inner dimensions must be equal for multiplication.")

    # Perform matrix multiplication using NumPy
    transformed_matrix = np.matmul(self.matrix1, self.matrix2)
    return transformed_matrix



def input_matrix():
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    matrix = []
    print("Enter the elements of the matrix:")
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(int(input(f"Enter element [{i+1}][{j+1}]: ")))
        matrix.append(row)
    numpy_matrix = np.array(matrix)
    return numpy_matrix

# Example usage
matrix1 = input_matrix()
matrix2 = input_matrix()

calculator = MatrixCalculator(matrix1, matrix2)

# # Addition
# result = calculator.add()
# if result is not None:
#   print("Sum:")
#   print(result)

# # Subtraction
# result = calculator.subtract()
# if result is not None:
#   print("Difference:")
#   print(result)

# # Multiplication
# result = calculator.multiply()
# if result is not None:
#   print("Product:")
#   print(result)
# #Divison
#   result = calculator.division()
# if result is not None:
#   print("Division:")
#   print(result)

# # Determinant (of matrix1)
# result = calculator.determinant()
# if result is not None:
#   print("Determinant of matrix1:")
#   print(result)

# #LU Factorization

# result = calculator.lu_decomposition()
# if result is not None:
#   print("LU factorization:")
#   print(result)

reult = calculator.LeastSquareSolution()
print(reult)

