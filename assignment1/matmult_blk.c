#define min(a, b)            \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b;       \
})

void matmult_blk(int m, int n, int k, double *A, double *B, double *C, int bs) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      C[i * n + j] = 0;
    }
  }

  for (int i1 = 0; i1 < m; i1 += bs) {
    for (int l1 = 0; l1 < k; l1 += bs) {
      for (int j1 = 0; j1 < n; j1 += bs) {
        // Sum the block multiplication
        // C[i1:i1 + bs, j1 :j1 + bs] += A[i1:i1 + bs, l1:l1 + bs]*B[l1:l1 + bs, j1:j1 + bs]
        for (int i2 = 0; i2 < min(m - i1, bs); i2++) {
          for (int l2 = 0; l2 < min(k - l1, bs); l2++) {
            for (int j2 = 0; j2 < min(n - j1, bs); j2++) {
              C[(i1 + i2) * n + j1 + j2] += A[(i1 + i2) * k + l1 + l2] * B[(l1 + l2) * n + j1 + j2];
            }
          }
        }
      }
    }
  }
}
