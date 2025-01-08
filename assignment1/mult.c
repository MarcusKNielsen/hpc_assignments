#include <stdio.h>

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


/*                 printf("i = %d \n", i);
                printf("j = %d \n", j);
                printf("l = %d \n", l);

                printf("a_idx = %d \n", a_idx);
                printf("b_idx = %d \n", b_idx);
                printf("c_idx = %d \n", c_idx);

                printf("A[i,l] = %f \n", A[a_idx]);
                printf("B[l,j] = %f \n", B[b_idx]);
                printf("C[i,j] = %f \n", C[c_idx]); */
