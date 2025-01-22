#ifndef MATMULT_CUBLAS_H
#define MATMULT_CUBLAS_H

extern "C" {
    void matmult_lib_offload(int m,int n,int k,double *A,double *B,double *C);
}

#endif
