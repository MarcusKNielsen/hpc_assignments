#include "matmult_asy_offload.h"

void matmult_asy_offload(int m,int n,int k,double *A,double *B,double *C) {

    #define SPLITS 8

    #pragma omp target enter data map(alloc: A[0:m*k])\
    map(alloc: B[0:k*n])\
    map(alloc: C[0:m*n])

    #pragma omp target data update(to: B[0:k*n])
    
    #pragma omp parallel for
    for (int s = 0; s < SPLITS; ++s) {

        int length = m / SPLITS;
        int lower  = s * length;
        int upper_A  = k * length;
        int upper_C  = n * length;

        #pragma omp target data update(to: A[lower:upper_A])

        #pragma omp target teams loop nowait\
        num_teams(length) thread_limit(32) 
        //map(to:A[lower:upper_A]) map(to:B) map(from:C[lower:upper_C]) 
        for (int i = lower; i < lower + length; ++i) {

            #pragma omp loop bind(parallel)
            for (int j=0; j < n; ++j) {
                double sum = 0;
                int c_idx = i * n + j;
                for (int l = 0; l < k; l++){

                    int b_idx = l * n + j;
                    int a_idx = i * k + l;
                    sum += A[a_idx] * B[b_idx];
                }
                C[c_idx] = sum;
            }

        }
        #pragma omp target data update(from: C[lower:upper_C])
    }
    #pragma omp taskwait
    #pragma omp target exit data map(release: A[0:m*k], B[0:m*k], C[0:m*n])
}