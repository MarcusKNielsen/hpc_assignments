#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "matmult_cublas.h"

void matmult_lib_offload(int m, int n, int k, double *A, double *B, double *C) {

  // cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Device pointers
  double *A_d, *B_d, *C_d;

  // Allocate device memory
  cudaMalloc((void **) &A_d, m * k * sizeof(double));
  cudaMalloc((void **) &B_d, k * n * sizeof(double));
  cudaMalloc((void **) &C_d, m * n * sizeof(double));

  // Copy data from host to device
  cudaMemcpy(A_d, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, k * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C, m * n * sizeof(double), cudaMemcpyHostToDevice);

  // Perform matrix-vector multiplication using cuBLAS
  const double alpha = 1.0;  // Scalar multiplier
  const double beta  = 0.0;  // Scalar added to result

  // cuBLAS compute matrix matrix product
  cublasDgemm(handle,
              CUBLAS_OP_N,  // transposition of the matrix A
              CUBLAS_OP_N,  // transposition of the matrix B
              n,            // Matrix dimension
              m,            // Matrix dimension
              k,            // Matrix dimension
              &alpha,       // Scalar alpha
              B_d,          // Matrix A on the device
              n,            // Leading dimension of the matrix A
              A_d,          // Matrix B on the device
              k,            // Leading dimension of the matrix B
              &beta,        // Scalar beta
              C_d,          // Result matrix C on the device
              n);           // Leading dimension of the matrix B


  // Copy result back to host
  cudaMemcpy(C, C_d, m * n * sizeof(double), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cublasDestroy(handle);

}
