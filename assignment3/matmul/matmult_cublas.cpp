#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "matmult_cublas.h"

void matmult_lib_offload(int m,int n,int k,double *A,double *B,double *C) {
    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Device pointers
    double *A_d, *B_d, *C_d;

    // Allocate device memory
    cudaMalloc((void **)&C_d, m * n * sizeof(double));
    cudaMalloc((void **)&A_d, m * k * sizeof(double));
    cudaMalloc((void **)&B_d, k * n * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(C_d, C, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, B, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, A, k * n * sizeof(double), cudaMemcpyHostToDevice);

    // Perform matrix-vector multiplication using cuBLAS
    const double alpha = 1.0; // Scalar multiplier
    const double beta = 0.0;  // Scalar added to result

    // cuBLAS uses column-major order, so we interpret the input matrix accordingly
    cublasDgemm(handle,
                CUBLAS_OP_N,  // No transposition of the matrix A
                CUBLAS_OP_N,  // No transposition of the matrix B
                m,            // Matrix dimension
                n,            // Matrix dimension
                k,            // Matrix dimension
                &alpha,       // Scalar alpha
                A_d,          // Matrix A on the device
                k,            // Leading dimension of the matrix A
                B_d,          // Matrix B on the device
                n,            // Leading dimension of the matrix B
                &beta,        // Scalar beta
                C_d,          // Result matrix C on the device
                n);           // Leading dimension of the matrix B

    // Copy result back to host
    cudaMemcpy(C, C_d, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(C_d);
    cudaFree(A_d);
    cudaFree(B_d);
    cublasDestroy(handle);
}
