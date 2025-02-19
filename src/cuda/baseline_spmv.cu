#include <cuda_runtime.h>
#include <stdio.h>

// CSR format sparse matrix structure for GPU
struct CSRMatrix {
    double* values;        // non-zero values
    int* col_indices;      // column indices for values
    int* row_ptr;         // pointers to start of each row
    int rows;
    int cols;
    int nnz;             // number of non-zero elements
};

// Error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        exit(1);
    }
}

// Baseline SpMV kernel implementation using CSR format.
// 
// Matrix format (CSR):
// - values: non-zero values
// - col_indices: column indices for each non-zero
// - row_ptr: indices into values/col_indices marking start of each row
// 
// Thread mapping:
// - One thread per row
// - Simple but not optimal for rows with many non-zeros
__global__ void spmv_csr_kernel(
    const float* __restrict__ values,      // Non-zero values
    const int* __restrict__ col_indices,   // Column indices
    const int* __restrict__ row_ptr,       // Row pointers
    const float* __restrict__ x,           // Input vector
    float* __restrict__ y,                 // Output vector
    const int num_rows                     // Number of rows
) {
    // Get global thread ID
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread handles a valid row
    if (row < num_rows) {
        // Get start and end positions for this row
        const int row_start = row_ptr[row];
        const int row_end = row_ptr[row + 1];
        
        // Compute dot product for this row
        float sum = 0.0f;
        for (int i = row_start; i < row_end; i++) {
            const int col = col_indices[i];
            sum += values[i] * x[col];
        }
        
        // Write result
        y[row] = sum;
    }
}

// Helper function to launch the kernel with appropriate grid/block size
extern "C" void launch_spmv(
    const float* values,
    const int* col_indices,
    const int* row_ptr,
    const float* x,
    float* y,
    const int num_rows
) {
    // Use 256 threads per block
    const int block_size = 256;
    const int num_blocks = (num_rows + block_size - 1) / block_size;
    
    // Launch kernel
    spmv_csr_kernel<<<num_blocks, block_size>>>(
        values, col_indices, row_ptr, x, y, num_rows
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}

// Host function to allocate GPU memory and copy data
CSRMatrix create_gpu_matrix(const double* h_values,
                           const int* h_col_indices,
                           const int* h_row_ptr,
                           const int rows,
                           const int cols,
                           const int nnz) {
    CSRMatrix d_matrix;
    d_matrix.rows = rows;
    d_matrix.cols = cols;
    d_matrix.nnz = nnz;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.values, nnz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.col_indices, nnz * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.row_ptr, (rows + 1) * sizeof(int)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.col_indices, h_col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.row_ptr, h_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    
    return d_matrix;
}

// Host function to free GPU memory
void free_gpu_matrix(CSRMatrix& matrix) {
    CHECK_CUDA_ERROR(cudaFree(matrix.values));
    CHECK_CUDA_ERROR(cudaFree(matrix.col_indices));
    CHECK_CUDA_ERROR(cudaFree(matrix.row_ptr));
}

// Main SpMV function that orchestrates the computation
extern "C" void spmv_cuda(const double* h_values,
                         const int* h_col_indices,
                         const int* h_row_ptr,
                         const double* h_x,
                         double* h_y,
                         const int rows,
                         const int cols,
                         const int nnz) {
    // Create GPU matrix
    CSRMatrix d_matrix = create_gpu_matrix(h_values, h_col_indices, h_row_ptr, rows, cols, nnz);
    
    // Allocate and copy input vector
    double* d_x;
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, cols * sizeof(double), cudaMemcpyHostToDevice));
    
    // Allocate output vector
    double* d_y;
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, rows * sizeof(double)));
    
    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    spmv_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_matrix.values,
        d_matrix.col_indices,
        d_matrix.row_ptr,
        d_x,
        d_y,
        rows
    );
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_y, d_y, rows * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Cleanup
    free_gpu_matrix(d_matrix);
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
}

// CUDA kernel for SpMV
__global__ void spmv_kernel(const double* values, 
                           const int* col_indices,
                           const int* row_ptr,
                           const double* x,
                           double* y,
                           const int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        double sum = 0.0;
        const int row_start = row_ptr[row];
        const int row_end = row_ptr[row + 1];
        
        // Compute dot product for this row
        for (int i = row_start; i < row_end; i++) {
            sum += values[i] * x[col_indices[i]];
        }
        
        y[row] = sum;
    }
}
