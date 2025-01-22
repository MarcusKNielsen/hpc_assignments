#ifndef MATVEC_CUBLAS_H
#define MATVEC_CUBLAS_

extern "C" {
    void matmult_lib_offload(int m,int n,int k,double *A,double *B,double *C);
}
#endif
