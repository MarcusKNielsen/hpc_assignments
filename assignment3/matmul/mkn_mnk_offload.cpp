//
// Created by seb-sti1 on 22/01/25.
//

#include "mkn_mnk_offload.h"

void matmult_mkn_offload(int m, int n, int k, double *A, double *B, double *C) {
#pragma omp target teams num_teams(m) thread_limit(32) distribute parallel for map(tofrom: C[0:m*n])
  for (int c_idx = 0; c_idx < m * n; c_idx++) {
    C[c_idx] = 0;
  }

#pragma omp target teams num_teams(m) thread_limit(32) distribute parallel for map(to: A[0:m*k], B[0:k*n]) map(tofrom: C[0:m*n])
  for (int i = 0; i < m; i++) {
    for (int l = 0; l < k; l++) {
      for (int j = 0; j < n; j++) {
        C[i * n + j] += A[i * k + l] * B[l * n + j];
      }
    }
  }
}

void matmult_mnk_offload(int m, int n, int k, double *A, double *B, double *C) {
#pragma omp target teams num_teams(m) thread_limit(32) distribute parallel for map(to: A[0:m*k], B[0:k*n]) map(tofrom: C[0:m*n])
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0;
      for (int l = 0; l < k; l++) {
        sum += A[i * k + l] * B[l * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}