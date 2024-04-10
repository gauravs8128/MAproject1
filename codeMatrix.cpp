#include <iostream>
#include <vector>
#include <limits>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace std;

// Function to get matrix dimensions from user
pair<int, int> getMatrixDimensions() {
    int rows, cols;
    cout << "Enter the number of rows: ";
    cin >> rows;
    cout << "Enter the number of columns: ";
    cin >> cols;
    return make_pair(rows, cols);
}

// Function to get matrix elements from user
vector<vector<double>> getMatrix(int rows, int cols) {
    vector<vector<double>> matrix(rows, vector<double>(cols));
    cout << "Enter matrix elements row-wise:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cin >> matrix[i][j];
        }
    }
    return matrix;
}

// Function to print a matrix
void printMatrix(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j

// Function to calculate the rank of a matrix
int matrixRank(const MatrixXd& matrix) {
    FullPivLU<MatrixXd> lu(matrix);
    return lu.rank();
}

// Function to perform Gauss-Jordan elimination
MatrixXd gaussJordan(const MatrixXd& matrix, const MatrixXd& rhs) {
    MatrixXd augmented(matrix.rows(), matrix.cols() + rhs.cols());
    augmented << matrix, rhs;

    int rows = augmented.rows();
    int cols = augmented.cols();

    for (int i = 0; i < rows; ++i) {
        // Find pivot row
        int pivotRow = i;
        for (int k = i + 1; k < rows; ++k) {
            if (abs(augmented(k, i)) > abs(augmented(pivotRow, i))) {
                pivotRow = k;
            }
        }

        // Swap current row with pivot row
        if (pivotRow != i) {
            augmented.row(i).swap(augmented.row(pivotRow));
        }

        // Make pivot element 1
        double pivot = augmented(i, i);
        for (int j = i; j < cols; ++j) {
            augmented(i, j) /= pivot;
        }

        // Make other elements in the column 0
        for (int k = 0; k < rows; ++k) {
            if (k != i) {
                double factor = augmented(k, i);
                for (int j = i; j < cols; ++j) {
                    augmented(k, j) -= factor * augmented(i, j);
                }
            }
        }
    }

    return augmented.block(0, matrix.cols(), matrix.rows(), rhs.cols());
}

// Function to perform LU factorization
tuple<MatrixXd, MatrixXd, MatrixXd> luFactorization(const MatrixXd& matrix) {
    PartialPivLU<MatrixXd> lu(matrix);
    MatrixXd lower = lu.matrixLU().triangularView<StrictlyLower>();
    MatrixXd upper = lu.matrixLU().triangularView<Upper>();
    MatrixXd permutation = lu.permutationP();

    return make_tuple(lower, upper, permutation);
}

// Function to calculate the determinant of a matrix
double matrixDeterminant(const MatrixXd& matrix) {
    return matrix.determinant();
}

// Function to calculate the transpose of a matrix
MatrixXd matrixTranspose(const MatrixXd& matrix) {
    return matrix.transpose();
}

// Function to calculate the column space of a matrix
MatrixXd columnSpace(const MatrixXd& matrix) {
    FullPivLU<MatrixXd> lu(matrix);
    return lu.matrixLU().block(0, 0, matrix.rows(), lu.rank());
}

int main() {
    // Example usage
    MatrixXd A(3, 3);
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    MatrixXd b(3, 1);
    b << 1,
         2,
         3;

    cout << "Rank of matrix A: " << matrixRank(A) << endl;

    cout << "Gauss-Jordan solution: " << endl;
    cout << gaussJordan(A, b) << endl;

    auto [L, U, P] = luFactorization(A);
    cout << "Lower triangular matrix L:" << endl << L << endl;
    cout << "Upper triangular matrix U:" << endl << U << endl;
    cout << "Permutation matrix P:" << endl << P << endl;

    cout << "Determinant of matrix A: " << matrixDeterminant(A) << endl;

    cout << "Transpose of matrix A:" << endl << matrixTranspose(A) << endl;

    cout << "Column space of matrix A:" << endl << columnSpace(A) << endl;

    return 0;
}
