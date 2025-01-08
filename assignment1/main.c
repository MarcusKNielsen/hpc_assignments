#include <stdio.h>
#include <stdlib.h>
#include "mult.h"



// Function to generate a random array of doubles
void generate_random_matrix(double *array, int size, double min, double max) {
    for (int i = 0; i < size; i++) {
        // Scale the random value to the [min, max] range
        array[i] = min + (max - min) * ((double)rand() / RAND_MAX);
    }
}

// Print a matrix in row major format
void print_matrix(double *matrix, int num_rows, int num_cols) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("%.2f ", matrix[i * num_cols + j]);
        }
        printf("\n");
    }
}


int main() {

    // Seed for the random number generator
    int SEED = 1;
    srand(SEED);

    // Range for random number generator
    double min = -5.0;
    double max = 5.0; 

    // Matrix dimensions
    int m = 4;
    int k = 2;
    int n = 3;


    // Allocate memory for matrix A
    double *A = (double *)malloc(m*k*sizeof(double));

    if (A == NULL) {
        perror("Memory allocation failed");
        return EXIT_FAILURE;
    }

    // Generate the random numbers for A
    generate_random_matrix(A, m*k, min, max);


    // Allocate memory for matrix B
    double *B = (double *)malloc(k*n*sizeof(double));

    if (B == NULL) {
        perror("Memory allocation failed");
        return EXIT_FAILURE;
    }

    // Generate the random numbers for B
    generate_random_matrix(B, k*n, min, max);


    // Print matrices
    printf("A = \n");
    print_matrix(A, m, k);
    printf("\n");

    printf("B = \n");
    print_matrix(B, k, n);
    printf("\n");

    // Allocate memory for matrix C
    double *C = (double *)malloc(m*n*sizeof(double));

    if (C == NULL) {
        perror("Memory allocation failed");
        return EXIT_FAILURE;
    }

    // Compute C = A*B
    matmult_nat(m,n,k,A,B,C);

    printf("C = \n");
    print_matrix(C, m, n);
    printf("\n");

    // Compute C = A*B using cblas
    matmult_lib(m,n,k,A,B,C);

    printf("C_cblas = \n");
    print_matrix(C, m, n);
    printf("\n");

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
