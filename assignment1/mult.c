#include <stdio.h>
#include <cblas.h>

// Function to compute product C=A*B using cblas
void matmult_lib(int m,int n,int k,double *A,double *B,double *C){

    // Parameters for cblas_dgemm
    double alpha = 1.0;
    double  beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,m, n, k, alpha, A, k, B, n, beta, C, n);
}

void matmult_nat(int m,int n,int k,double *A,double *B,double *C){

    // These two loops iterates over indexes of C: C[i,j]
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            int c_idx = i*n+j; // row-major indexing: "C[i,j] = C[c_idx]"
            C[c_idx] = 0;

            // This loop computes: C[i,j] = sum_l A[i,l]*B[l,j]
            for (int l=0; l<k; l++){
                int a_idx = i*k+l; // row-major indexing: "A[i,l] = A[a_idx]"
                int b_idx = l*n+j; // row-major indexing: "B[l,j] = B[b_idx]"

                C[c_idx] += A[a_idx]*B[b_idx];
            }


        }
    }
}

// Matrix multiplication with m as the outer loop, n as the next one and k 
// as the inner loop. We already have this function (matmult_nat), so
// we simply call this function.
void matmult_mnk(int m,int n,int k,double *A,double *B,double *C){
    matmult_nat(m, n, k, A, B, C);
}

void matmult_mkn(int m,int n,int k,double *A,double *B,double *C){
    // Intialize C-matrix to 0.
    for (int c_idx = 0; c_idx < m * n; c_idx++) {
        C[c_idx] = 0;
    }

    // These two loops iterates over indexes of C: C[i,j]
    for (int i=0; i<m; i++){
        for (int l=0; l<k; l++){
            int a_idx = i*k+l; // row-major indexing: "A[i,l] = A[a_idx]"

            // This loop computes: C[i,j] = sum_l A[i,l]*B[l,j]
            for (int j=0; j<n; j++){
                int c_idx = i*n+j; // row-major indexing: "C[i,j] = C[c_idx]"
                int b_idx = l*n+j; // row-major indexing: "B[l,j] = B[b_idx]"

                C[c_idx] += A[a_idx]*B[b_idx];
            }
        }
    }
}


void matmult_kmn(int m,int n,int k,double *A,double *B,double *C){
    // Intialize C-matrix to 0.
    for (int c_idx = 0; c_idx < m * n; c_idx++) {
        C[c_idx] = 0;
    }

    // These two loops iterates over indexes of C: C[i,j]
    for (int l=0; l<k; l++){
        for (int i=0; i<m; i++){
            int a_idx = i*k+l; // row-major indexing: "A[i,l] = A[a_idx]"

            // This loop computes: C[i,j] = sum_l A[i,l]*B[l,j]
            for (int j=0; j<n; j++){
                int c_idx = i*n+j; // row-major indexing: "C[i,j] = C[c_idx]"
                int b_idx = l*n+j; // row-major indexing: "B[l,j] = B[b_idx]"

                C[c_idx] += A[a_idx]*B[b_idx];
            }
        }
    }
}


void matmult_knm(int m,int n,int k,double *A,double *B,double *C){
    // Intialize C-matrix to 0.
    for (int c_idx = 0; c_idx < m * n; c_idx++) {
        C[c_idx] = 0;
    }

    // These two loops iterates over indexes of C: C[i,j]
    for (int l=0; l<k; l++){
        for (int j=0; j<n; j++){
            int b_idx = l*n+j; // row-major indexing: "B[l,j] = B[b_idx]"

            // This loop computes: C[i,j] = sum_l A[i,l]*B[l,j]
            for (int i=0; i<m; i++){
                int c_idx = i*n+j; // row-major indexing: "C[i,j] = C[c_idx]"
                int a_idx = i*k+l; // row-major indexing: "A[i,l] = A[a_idx]"

                C[c_idx] += A[a_idx]*B[b_idx];
            }
        }
    }
}



void matmult_nkm(int m,int n,int k,double *A,double *B,double *C){
    // Intialize C-matrix to 0.
    for (int c_idx = 0; c_idx < m * n; c_idx++) {
        C[c_idx] = 0;
    }

    // These two loops iterates over indexes of C: C[i,j]
    for (int j=0; j<n; j++){
        for (int l=0; l<k; l++){
            int b_idx = l*n+j; // row-major indexing: "B[l,j] = B[b_idx]"

            // This loop computes: C[i,j] = sum_l A[i,l]*B[l,j]
            for (int i=0; i<m; i++){
                int c_idx = i*n+j; // row-major indexing: "C[i,j] = C[c_idx]"
                int a_idx = i*k+l; // row-major indexing: "A[i,l] = A[a_idx]"

                C[c_idx] += A[a_idx]*B[b_idx];
            }
        }
    }
}


void matmult_nmk(int m,int n,int k,double *A,double *B,double *C){

    // These two loops iterates over indexes of C: C[i,j]
    for (int j=0; j<n; j++){
        for (int i=0; i<m; i++){
            int c_idx = i*n+j; // row-major indexing: "C[i,j] = C[c_idx]"
            C[c_idx] = 0;

            // This loop computes: C[i,j] = sum_l A[i,l]*B[l,j]
            for (int l=0; l<k; l++){
                int a_idx = i*k+l; // row-major indexing: "A[i,l] = A[a_idx]"
                int b_idx = l*n+j; // row-major indexing: "B[l,j] = B[b_idx]"

                C[c_idx] += A[a_idx]*B[b_idx];
            }
        }
    }
}



/*                 printf("i = %d \n", i);
                printf("j = %d \n", j);
                printf("l = %d \n", l);

                printf("a_idx = %d \n", a_idx);
                printf("b_idx = %d \n", b_idx);
                printf("c_idx = %d \n", c_idx);

                printf("A[i,l] = %f \n", A[a_idx]);
                printf("B[l,j] = %f \n", B[b_idx]);
                printf("C[i,j] = %f \n", C[c_idx]); */
