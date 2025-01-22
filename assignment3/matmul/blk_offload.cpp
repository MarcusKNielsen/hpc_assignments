#include "blk_offload.h"

#define min(a, b)            \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b;       \
})

#define BLK 100

void matmult_blk_offload(int m, int n, int k, double *A, double *B, double *C){
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
        }
    }

    #pragma omp target teams num_teams(m) thread_limit(64) \
    distribute parallel for \
    map(to: A[0:m*k], B[0:k*n]) map(tofrom: C[0:m*n])
    for (int i1 = 0; i1 < m; i1 += BLK) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                for(int i2 = 0; i2 < min(m-BLK,BLK); i2++){
                    C[(i1 + i2) * n + j] += A[(i1 + i2) * k + l] * B[l * n + j];
                }
            }
        }
    }
}