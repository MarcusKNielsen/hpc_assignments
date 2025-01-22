#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

void matvec_cublas(int n, int m, double *mat, double *vec, double *res) {
    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Device pointers
    double *d_mat, *d_vec, *d_res;

    // Allocate device memory
    cudaMalloc((void **)&d_mat, n * m * sizeof(double));
    cudaMalloc((void **)&d_vec, m * sizeof(double));
    cudaMalloc((void **)&d_res, n * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_mat, mat, n * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec, m * sizeof(double), cudaMemcpyHostToDevice);

    // Perform matrix-vector multiplication using cuBLAS
    const double alpha = 1.0; // Scalar multiplier
    const double beta = 0.0;  // Scalar added to result

    // cuBLAS uses column-major order, so we interpret the input matrix accordingly
    cublasDgemv(handle,
                CUBLAS_OP_N,  // No transposition of the matrix
                n,            // Number of rows in the matrix
                m,            // Number of columns in the matrix
                &alpha,       // Scalar alpha
                d_mat,        // Matrix on the device
                n,            // Leading dimension of the matrix (number of rows)
                d_vec,        // Vector on the device
                1,            // Stride between elements of the vector
                &beta,        // Scalar beta
                d_res,        // Result vector on the device
                1);           // Stride between elements of the result vector

    // Copy result back to host
    cudaMemcpy(res, d_res, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_res);
    cublasDestroy(handle);
}
