#include <cblas.h>
#include <omp.h>
#include "matmult_mkn_omp.h"

// Function to compute product C=A*B using cblas
void matmult_lib(int m,int n,int k,double *A,double *B,double *C){

    // Parameters for cblas_dgemm
    double alpha = 1.0;
    double  beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,m, n, k, alpha, A, k, B, n, beta, C, n);
}

void matmult_mkn_omp(int m,int n,int k,double *A,double *B,double *C){
    // Intialize C-matrix to 0.
    #pragma omp parallel for
    for (int c_idx = 0; c_idx < m * n; c_idx++) {
        C[c_idx] = 0;
    }

    // These two loops iterates over indexes of C: C[i,j]
    #pragma omp parallel
    {
        // Each thread will pick up work for each `i`
        #pragma omp for
        for (int i = 0; i < m; i++) {
            for (int l = 0; l < k; l++) {
                int a_idx = i * k + l; // row-major indexing: "A[i,l] = A[a_idx]"

                for (int j = 0; j < n; j++) {
                    int c_idx = i * n + j; // row-major indexing: "C[i,j] = C[c_idx]"
                    int b_idx = l * n + j; // row-major indexing: "B[l,j] = B[b_idx]"

                    // Since each thread works on a unique `i`, no race condition
                    C[c_idx] += A[a_idx] * B[b_idx];
                }
            }
        }
    }
}