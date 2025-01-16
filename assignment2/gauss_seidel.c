/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include "gauss_seidel.h"


// Solve the full gauss-seidel in parallel. Return the number of iterations used.
void solve_gauss_seidel(double ***u, double ***f, int N, int max_it) {
  int k = 0;

  while (k < max_it) {
    gauss_seidel(u, f, N);
    k++;
  }
}



// Updates u for each element of the 3D matrix. Returns the norm between two iterations.
void gauss_seidel(double ***u, double ***f, int N) {
  double delta_squared = (2.0 / (N + 1));
  delta_squared = delta_squared * delta_squared;
  double one_sixth = 1.0 / 6.0;

 #pragma omp parallel
 {
 #pragma omp parallel for schedule(static,1) ordered(2)
  for (int i  = 1; i < N + 1; i++) {
    for (int j = 1; j < N + 1; j++) {
     #pragma omp ordered depend(sink: i-1,j) depend(sink: i,j-1)
      for (int k = 1; k < N + 1; k++) {
        u[i][j][k] = one_sixth * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + delta_squared * f[i][j][k]);
      }
     #pragma omp ordered depend(source)
    }
  }
 }
}