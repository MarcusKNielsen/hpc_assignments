#include "blk_offload.h"

#define min(a, b)            \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b;       \
})

#define BLK 35

void matmult_blk_offload(int m, int n, int k, double *A, double *B, double *C){
    /*
    #pragma omp parallel for default(none) \
    shared(m, n, C)
    for (int c_idx = 0; c_idx < m * n; c_idx++) {
        C[c_idx] = 0;
    }

    #pragma omp target teams num_teams(m) thread_limit(64) \
    distribute parallel for \
    map(to: A[0:m*k], B[0:k*n]) map(tofrom: C[0:m*n]) \
    collapse(2)
    for (int i1 = 0; i1 < m; i1 += BLK) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                int bound = min(m-BLK,BLK);
                for(int i2 = 0; i2 < bound; i2++){
                    C[(i1 + i2) * n + j] += A[(i1 + i2) * k + l] * B[l * n + j];
                }
            }
        }
    }
    */
    #pragma omp target teams num_teams(m) thread_limit(64) \
    distribute parallel for \
    map(to: A[0:m*k], B[0:k*n]) map(tofrom: C[0:m*n]) \
    collapse(2)
    for(int i = 0; i < BLK; i += BLK){
        for(int j = 0; j < n; j++){
            if(i + BLK - 1 < m){
                double sum[BLK] = {0};
                for(int i1 = 0; i1 < BLK; i1++){
                    for(int l = 0; l < k; l++){
                        sum[i1] += A[(i + i1) * k + l] * B[l * n + j];
                    } 
                    C[(i + i1) * n + j] = sum;
                }
            }
            else {
                double sum[BLK] = {0};
                int bound = m - BLK;
                for(int i1 = 0; i1 < bound; i1++){
                    for(int l = 0; l < k; l++){
                        sum[i1] += A[(i + i1) * k + l] * B[l * n + j];
                    } 
                    C[(i + i1) * n + j] = sum;
                }
            }
        }
    }
}