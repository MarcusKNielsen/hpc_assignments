#ifndef MATMULT_ASY_OFFLOAD_H
#define MATMULT_ASY_OFFLOAD_H

extern "C" {
    void matmult_asy_offload(int m,int n,int k,double *A,double *B,double *C);
}

#endif