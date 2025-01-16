/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include "gauss_seidel.h"

// Solve the full gauss-seidel. Return the number of iterations used.
int solve_gauss_seidel(double ***u, double ***f, int N, int max_it, double threshold) {
  int k = 0;
  double d = INFINITY;
  double threshold_squared = threshold * threshold;

  while ((d > threshold_squared) && (k < max_it)) {
    d = gauss_seidel(u, f, N);
    k++;
  }

  return k;
}

// Updates u for each element of the 3D matrix. Returns the norm between two iterations.
double gauss_seidel(double ***u, double ***f, int N) {
  double delta_squared = (2.0 / (N + 1));
  delta_squared = delta_squared * delta_squared;
  double one_sixth = 1.0 / 6.0;
  double norm = 0.0;

  for (int i  = 1; i < N + 1; i++) {
    for (int j = 1; j < N + 1; j++) {
      for (int k = 1; k < N + 1; k++) {

        double temp_u =  u[i][j][k];
        u[i][j][k] = one_sixth * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + delta_squared * f[i][j][k]);
        temp_u = u[i][j][k] - temp_u;

        norm += temp_u * temp_u;
      }
    }
  }

  return norm / N / N / N;
}





// Solve the full gauss-seidel in parallel. Return the number of iterations used.
void parallel_solve_gauss_seidel(double ***u, double ***f, int N, int max_it) {
  int k = 0;

  while (k < max_it) {
    gauss_seidel(u, f, N);
    k++;
  }
}



// Updates u for each element of the 3D matrix. Returns the norm between two iterations.
void parallel_gauss_seidel(double ***u, double ***f, int N) {
  double delta_squared = (2.0 / (N + 1));
  delta_squared = delta_squared * delta_squared;
  double one_sixth = 1.0 / 6.0;

  for (int i  = 1; i < N + 1; i++) {
    for (int j = 1; j < N + 1; j++) {
      for (int k = 1; k < N + 1; k++) {
        u[i][j][k] = one_sixth * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + delta_squared * f[i][j][k]);
      }
    }
  }
}