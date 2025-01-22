#ifndef MATMUL__BLK_OFFLOAD_H_
#define MATMUL__BLK_OFFLOAD_H_

extern "C" {

void matmult_blk_offload(int m, int n, int k, double *A, double *B, double *C);

}
#endif 