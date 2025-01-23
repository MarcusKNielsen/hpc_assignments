#include "matmult_asy_offload.h"

#ifndef SPLITS
#define SPLITS 5
#endif

void matmult_asy_offload(int m, int n, int k, double *A, double *B, double *C) {
  int length = m / SPLITS;
#pragma omp target enter data map(alloc: A[0:m*k], B[0:k*n], C[0:m*n])
#pragma omp target update to(B[0:k*n])
  for (int s = 0; s < SPLITS; ++s) {
    int lower = s * length;
#pragma omp target update to(A[k*lower:k*length])

#pragma omp target teams loop nowait \
        num_teams(length) thread_limit(32)
    for (int i = lower; i < lower + length; i++) {
#pragma omp loop bind(parallel)
      for (int j = 0; j < n; j++) {
        double sum = 0;
        int c_idx = i * n + j;
        for (int l = 0; l < k; l++) {

          int b_idx = l * n + j;
          int a_idx = i * k + l;
          sum += A[a_idx] * B[b_idx];
        }
        C[c_idx] = sum;
      }

    }
#pragma omp target update from(C[n*lower:n*length])
  }
#pragma omp taskwait
#pragma omp target exit data map(release: A[0:m*k], B[0:k*n], C[0:m*n])
}