/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include <time.h>

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#define N_DEFAULT 100

void initialize_data(double ***u, double ***f, int N) {
  for (int i = 0; i < N + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      for (int k = 0; k < N + 2; k++) {
        u[i][j][k] = 0.0;
        f[i][j][k] = 0.0;
      }
    }
  }

  for (int i = 0; i < N + 2; i++) {
    for (int k = 0; k < N + 2; k++) {
      u[i][0][k] = 0;
      u[i][N + 1][k] = 20;
    }
  }

  for (int j = 0; j < N + 2; j++) {
    for (int k = 0; k < N + 2; k++) {
      u[0][j][k] = 20;
      u[N + 1][j][k] = 20;
    }
  }

  for (int i = 0; i < N + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      u[i][j][0] = 20;
      u[i][j][N + 1] = 20;
    }
  }

  int i_min = 0;
  int j_min = 0;
  int k_min = (int) ((N + 2.0) / 3.0);
  int i_max = (int) ((N + 2.0) * 1.5 / 8.0);
  int j_max = (int) ((N + 2.0) / 4.0);
  int k_max = (int) ((N + 2.0) / 2.0);

  for (int i = i_min; i < i_max; i++) {
    for (int j = j_min; j < j_max; j++) {
      for (int k = k_min; k < k_max; k++) {
        f[i][j][k] = 200;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int N = N_DEFAULT;
  int iter_max = 1000;
  double tolerance;
  double start_T;
  int output_type = 0;
  char *output_prefix = "poisson_res";
  char *output_ext = "";
  char output_filename[FILENAME_MAX];

  /* get the parameters from the command line */
  N = atoi(argv[1]);    // grid size
  iter_max = atoi(argv[2]);  // max. no. of iterations
  tolerance = atof(argv[3]);  // tolerance
  start_T = atof(argv[4]);  // start T for all inner grid points
  if (argc == 6) {
    output_type = atoi(argv[5]);  // output type
  }

  double allocation_t, initialize_t, compute_t = 0;

  double ***u = NULL;
  double ***u2 = NULL;
  double ***f = NULL;

  allocation_t -= (double) clock() / CLOCKS_PER_SEC;
  // allocate memory
  if ((u = malloc_3d(N + 2, N + 2, N + 2)) == NULL) {
    perror("array u: allocation failed");
    exit(-1);
  }

#ifdef _GAUSS_SEIDEL
  if ((u2 = malloc_3d(N + 2, N + 2, N + 2)) == NULL) {
    perror("array u: allocation failed");
    exit(-1);
  }
#endif

  if ((f = malloc_3d(N + 2, N + 2, N + 2)) == NULL) {
    perror("array f: allocation failed");
    exit(-1);
  }

  allocation_t += (double) clock() / CLOCKS_PER_SEC;

  initialize_t -= (double) clock() / CLOCKS_PER_SEC;
  initialize_data(u, f, N);
  initialize_t += (double) clock() / CLOCKS_PER_SEC;

  compute_t -= (double) clock() / CLOCKS_PER_SEC;
  int iter = 0;
#ifdef _JACOBI
  iter = solve_jacobi(u, u2, f, N, iter_max, tolerance);
#endif
#ifdef _GAUSS_SEIDEL
  iter = solve_gauss_seidel(u, f, N, iter_max, tolerance);
#endif
  compute_t += (double) clock() / CLOCKS_PER_SEC;

  printf("%f, %f, %f, %d\n", allocation_t, initialize_t, compute_t, iter * N * N * N);

  // dump  results if wanted
  switch (output_type) {
    case 0:
      // no output at all
      break;
    case 3: output_ext = ".bin";
      sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
      fprintf(stderr, "Write binary dump to %s: ", output_filename);
      print_binary(output_filename, N, u);
      break;
    case 4: output_ext = ".vtk";
      sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
      fprintf(stderr, "Write VTK file to %s: ", output_filename);
      print_vtk(output_filename, N, u);
      break;
    default: fprintf(stderr, "Non-supported output type!\n");
      break;
  }

  // de-allocate memory
  free_3d(u);

  return (0);
}
