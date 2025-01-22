#ifndef MATVEC_CUBLAS_H
#define MATVEC_CUBLAS_H

extern "C" {
    void matvec_cublas(int m,int n,int k,double *A,double *B,double *C);
}
#endif
