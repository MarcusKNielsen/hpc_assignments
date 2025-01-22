/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <stddef.h>
#include <stdio.h>
#include "jacobi_part7.h"

int solve_jacobi(double ***U_new_d0,
                 double ***U_old_d0,
                 double ***F_d0,
                 double ***U_new_d1,
                 double ***U_old_d1,
                 double ***F_d1,
                 int N,
                 int max_it,
                 double threshold) {

  for (int iter = 0; iter < max_it; iter++) {

    jacobi(U_new_d0, U_old_d0, F_d0, U_new_d1, U_old_d1, F_d1, N);

    double ***tmp_d0 = U_old_d0;
    U_old_d0 = U_new_d0;
    U_new_d0 = tmp_d0;

    double ***tmp_d1 = U_old_d1;
    U_old_d1 = U_new_d1;
    U_new_d1 = tmp_d1;
  }

  return max_it;
}

//#pragma omp declare target
void jacobi(double ***U_new_d0,
            double ***U_old_d0,
            double ***F_d0,
            double ***U_new_d1,
            double ***U_old_d1,
            double ***F_d1,
            int N) {
  double scale = 1.0 / 6.0;
  double delta_squared = 2.0 / (N + 1);
  delta_squared = delta_squared * delta_squared;
  // from i = 1 to ((N + 2) / 2) - 1 (excluded)
#pragma omp target teams loop is_device_ptr(U_new_d0, U_old_d0, F_d0, U_new_d1, U_old_d1, F_d1) \
    num_teams(N/2) thread_limit(32) collapse(2) device(0) nowait
  for (size_t i = 1; i < ((N + 2) / 2) - 1; i++) {
    for (size_t j = 1; j <= N; j++) {
#pragma omp loop bind(parallel)
      for (size_t k = 1; k <= N; k++) {
        U_new_d0[i][j][k] = scale * (
            U_old_d0[i - 1][j][k] +
                U_old_d0[i + 1][j][k] +
                U_old_d0[i][j - 1][k] +
                U_old_d0[i][j + 1][k] +
                U_old_d0[i][j][k - 1] +
                U_old_d0[i][j][k + 1] +
                delta_squared * F_d0[i][j][k]);
      }
    }
  }
  // do ((N + 2) / 2) - 1 (it uses shared mem from device1)
  size_t i = ((N + 2) / 2) - 1;
#pragma omp target teams loop is_device_ptr(U_new_d0, U_old_d0, F_d0, U_new_d1, U_old_d1, F_d1) \
    num_teams(N) thread_limit(32) device(0) nowait
  for (size_t j = 1; j <= N; j++) {
#pragma omp loop bind(parallel)
    for (size_t k = 1; k <= N; k++) {
      U_new_d0[i][j][k] = scale * (
          U_old_d0[i - 1][j][k] +
              U_old_d1[0][j][k] +
              U_old_d0[i][j - 1][k] +
              U_old_d0[i][j + 1][k] +
              U_old_d0[i][j][k - 1] +
              U_old_d0[i][j][k + 1] +
              delta_squared * F_d0[i][j][k]);
    }
  }
  // do ((N + 2) / 2) (it uses shared mem from device 0)
#pragma omp target teams loop is_device_ptr(U_new_d0, U_old_d0, F_d0, U_new_d1, U_old_d1, F_d1) \
    num_teams(N) thread_limit(32) device(1) nowait
  for (size_t j = 1; j <= N; j++) {
#pragma omp loop bind(parallel)
    for (size_t k = 1; k <= N; k++) {
      U_new_d1[0][j][k] = scale * (
          U_old_d0[((N + 2) / 2) - 1][j][k] +
              U_old_d1[0 + 1][j][k] +
              U_old_d1[0][j - 1][k] +
              U_old_d1[0][j + 1][k] +
              U_old_d1[0][j][k - 1] +
              U_old_d1[0][j][k + 1] +
              delta_squared * F_d1[0][j][k]);
    }
  }
  // from ((N + 2) / 2) + 1 to N + 2 - 1 (excluded)
#pragma omp target teams loop is_device_ptr(U_new_d0, U_old_d0, F_d0, U_new_d1, U_old_d1, F_d1) \
    num_teams(N/2) thread_limit(32) collapse(2) device(1) nowait
  for (size_t i = 1; i < ((N + 2) / 2) - 1; i++) {
    for (size_t j = 1; j <= N; j++) {
#pragma omp loop bind(parallel)
      for (size_t k = 1; k <= N; k++) {
        U_new_d1[i][j][k] = scale * (
            U_old_d1[i - 1][j][k] +
                U_old_d1[i + 1][j][k] +
                U_old_d1[i][j - 1][k] +
                U_old_d1[i][j + 1][k] +
                U_old_d1[i][j][k - 1] +
                U_old_d1[i][j][k + 1] +
                delta_squared * F_d1[i][j][k]);
      }
    }
  }
#pragma omp taskwait
}
//#pragma omp end declare target
