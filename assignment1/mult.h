#ifndef MULT_H
#define MULT_H

void matmult_lib(int m,int n,int k,double *A,double *B,double *C);
void matmult_nat(int m, int n, int k, double *A, double *B, double *C);
void matmult_mnk(int m,int n,int k,double *A,double *B,double *C);
void matmult_mkn(int m,int n,int k,double *A,double *B,double *C);
void matmult_nmk(int m,int n,int k,double *A,double *B,double *C);
void matmult_nkm(int m,int n,int k,double *A,double *B,double *C);
void matmult_kmn(int m,int n,int k,double *A,double *B,double *C);
void matmult_knm(int m,int n,int k,double *A,double *B,double *C);
#endif
