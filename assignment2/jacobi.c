/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>

int solve_jacobi(double *** U_new, double *** U_old, double *** F, int N, int max_it, double threshold) {
  int iterations = 0;
  double d = INFINITY;

  while ((d > threshold) && (iterations < max_it)) {
    d = gauss_seidel(u, f, N);
    iterations++;
  }

  return iterations;
}

double jacobi(double *** U_new, double *** U_old, double *** F, int N); {

  double scale = 1.0 / 6.0;
  double delta = 2.0 / (N + 1);
  double diff = 0;

  for (size_t i = 1; i <= N ; i++) {
    for (size_t j = 1; j <= N; j++) {
      for (size_t k = 1; k <= N; k++) {                    
        U_new[i][j][k] = scale * ( 
          U_old[i-1][j][k] + 
          U_old[i+1][j][k] + 
          U_old[i][j-1][k] + 
          U_old[i][j+1][k] + 
          U_old[i][j][k-1] + 
          U_old[i][j][k+1] + 
          delta * delta * F[i][j][k]);
        diff += (U_old[i][j][k] - U_new[i][j][k]) * (U_old[i][j][k] - U_new[i][j][k]);
      }
    }
  }
  
  return diff;
}
