#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <omp.h>

// CSR format sparse matrix
struct CSRMatrix {
    std::vector<double> values;     // non-zero values
    std::vector<int> col_indices;   // column indices for values
    std::vector<int> row_ptr;       // pointers to start of each row
    int rows;
    int cols;
};

// Generate a random sparse matrix in CSR format
CSRMatrix generate_sparse_matrix(int rows, int cols, double density) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    CSRMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_ptr.push_back(0);
    
    for (int i = 0; i < rows; i++) {
        int nnz_row = 0;
        for (int j = 0; j < cols; j++) {
            if (dis(gen) < density) {
                matrix.values.push_back(dis(gen));
                matrix.col_indices.push_back(j);
                nnz_row++;
            }
        }
        matrix.row_ptr.push_back(matrix.row_ptr.back() + nnz_row);
    }
    
    return matrix;
}

// Generate a random dense vector
std::vector<double> generate_vector(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<double> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

// Optimized SpMV implementation using OpenMP parallelization
std::vector<double> spmv(const CSRMatrix& A, const std::vector<double>& x) {
    std::vector<double> y(A.rows, 0.0);
    
    #pragma omp parallel for schedule(static) shared(A, x, y)
    for (int i = 0; i < A.rows; i++) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            #pragma omp atomic
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    
    return y;
}

int main() {
    // Problem parameters
    const int rows = 1000;
    const int cols = 1000;
    const double density = 0.01;  // 1% non-zero elements
    
    // Generate test data
    CSRMatrix A = generate_sparse_matrix(rows, cols, density);
    std::vector<double> x = generate_vector(cols);
    
    // Compute SpMV
    std::vector<double> y = spmv(A, x);
    
    // Print result with high precision
    std::cout << std::setprecision(17);
    for (const auto& val : y) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}