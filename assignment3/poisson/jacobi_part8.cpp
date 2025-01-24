/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <stddef.h>
#include "jacobi_part8.h"
#include "math.h"

int solve_jacobi(double ***U_new, double ***U_old, double ***F, int N, int max_it, double threshold) {
int iterations = 0;
double d = INFINITY;
double threshold_squared = threshold * threshold;

#pragma omp target data \
    map(to:U_old[:N+2][:N+2][:N+2], F[:N+2][:N+2][:N+2]) map(tofrom:U_new[:N+2][:N+2][:N+2])
  {
    while ((iterations < max_it)) {
      d = jacobi(U_new, U_old, F, N);

      double ***tmp = U_old;
      U_old = U_new;
      U_new = tmp;

      iterations++;
    }
  }
  return iterations;
}

double jacobi(double ***U_new, double ***U_old, double ***F, int N) {
  double scale = 1.0 / 6.0;
  double delta_squared = 2.0 / (N + 1);
  delta_squared = delta_squared * delta_squared;
  double diff = 0;

#pragma omp target teams distribute parallel for collapse(3) reduction(+:diff)
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
        diff += (U_old[i][j][k] - U_new[i][j][k]) * (U_old[i][j][k] - U_new[i][j][k]);
      }
    }
  }
// barrier (implied)

  return diff / N / N / N;
}

