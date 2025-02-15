#include "blk_offload.h"

#ifndef BLK
#define BLK 35
#endif

void matmult_blk_offload(int m, int n, int k, double *A, double *B, double *C) {

#pragma omp target teams \
    distribute parallel for \
    map(to: A[0:m*k], B[0:k*n]) map(tofrom: C[0:m*n]) \
    collapse(2)
  for (int i = 0; i < m; i += BLK) {
    for (int j = 0; j < n; j++) {
      if (i + BLK - 1 < m) {
        double sum[BLK] = {0};
        for (int l = 0; l < k; l++) {
          for (int i1 = 0; i1 < BLK; i1++) {
            sum[i1] += A[(i + i1) * k + l] * B[l * n + j];
          }
        }
        for (int i1 = 0; i1 < BLK; i1++) {
          C[(i + i1) * n + j] = sum[i1];
        }
      } else {
        for (int i1 = 0; i1 < m - i; i1++) {
          double sum = 0;
          for (int l = 0; l < k; l++) {
            sum += A[(i + i1) * k + l] * B[l * n + j];
          }
          C[(i + i1) * n + j] = sum;
        }
      }
    }
  }
}