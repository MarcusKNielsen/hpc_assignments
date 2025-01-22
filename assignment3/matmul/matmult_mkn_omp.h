#ifndef MALTMULT_MKN_OMP_H
#define MALTMULT_MKN_OMP_H

extern "C" {
    void matmult_lib(int m,int n,int k,double *A,double *B,double *C);
    void matmult_mkn_omp(int m,int n,int k,double *A,double *B,double *C);
}

#endif
