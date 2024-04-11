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
      print("Matrices must have the same dimensions for addition.")
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
  import numpy as np

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

# Example usage
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

calculator = MatrixCalculator(matrix1, matrix2)

# Addition
result = calculator.add()
if result is not None:
  print("Sum:")
  print(result)

# Subtraction
result = calculator.subtract()
if result is not None:
  print("Difference:")
  print(result)

# Multiplication
result = calculator.multiply()
if result is not None:
  print("Product:")
  print(result)
#Divison
  result = calculator.division()
if result is not None:
  print("Division:")
  print(result)

# Determinant (of matrix1)
result = calculator.determinant()
if result is not None:
  print("Determinant of matrix1:")
  print(result)



