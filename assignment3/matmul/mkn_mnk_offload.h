//
// Created by seb-sti1 on 22/01/25.
//

#ifndef MATMUL__MKN_OFFLOAD_H_
#define MATMUL__MKN_OFFLOAD_H_

void matmult_mkn_offload(int m, int n, int k, double *A, double *B, double *C);

void matmult_mnk_offload(int m, int n, int k, double *A, double *B, double *C);

#endif //MATMUL__MKN_OFFLOAD_H_
