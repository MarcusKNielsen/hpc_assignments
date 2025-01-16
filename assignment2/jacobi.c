/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <stddef.h>
#include "jacobi.h"

int solve_jacobi(double ***U_new, double ***U_old, double ***F, int N, int max_it, double threshold) {
  for (int iter = 0; iter < max_it; iter++) {
    jacobi(U_new, U_old, F, N);
    double ***tmp = U_old;
    U_old = U_new;
    U_new = tmp;
  }

  return max_it;
}

void jacobi(double ***U_new, double ***U_old, double ***F, int N) {

  double scale = 1.0 / 6.0;
  double delta_squared = 2.0 / (N + 1);
  delta_squared = delta_squared * delta_squared;

#pragma omp for schedule(static)
  for (size_t i = 1; i <= N; i++) {
    for (size_t j = 1; j <= N; j++) {
      for (size_t k = 1; k <= N; k++) {
        U_new[i][j][k] = scale * (
            U_old[i - 1][j][k] +
                U_old[i + 1][j][k] +
                U_old[i][j - 1][k] +
                U_old[i][j + 1][k] +
                U_old[i][j][k - 1] +
                U_old[i][j][k + 1] +
                delta_squared * F[i][j][k]);
      }
    }
  }
// barrier (implied)
}