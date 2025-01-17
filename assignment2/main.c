/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include <math.h>
#include <time.h>
#include <omp.h>

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#define N_DEFAULT 100

#define M_PI 3.14159265358979323846

void initialize_test_data(double ***u, double ***f, int N) {
  for (int i = 0; i < N + 2; i++) {
    double x = -1.0 + ((2.0 * i) / (N + 2));
    for (int j = 0; j < N + 2; j++) {
      double y = -1.0 + ((2.0 * j) / (N + 2));
      for (int k = 0; k < N + 2; k++) {
        double z = -1.0 + ((2.0 * k) / (N + 2));
        u[i][j][k] = 0.0;
        f[i][j][k] = 3 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
      }
    }
  }
}

void check_test_data(double ***u, int N, double tolerance) {
  for (int i = 0; i < N + 2; i++) {
    double x = -1.0 + ((2.0 * i) / (N + 2));
    for (int j = 0; j < N + 2; j++) {
      double y = -1.0 + ((2.0 * j) / (N + 2));
      for (int k = 0; k < N + 2; k++) {
        double z = -1.0 + ((2.0 * k) / (N + 2));
        double error = fabs(u[i][j][k] - sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z));
        if (error > tolerance) {
          printf("%d %d %d: %f\n", i, j, k, error);
        }
      }
    }
  }
}

void initialize_border(double ***u, int N) {
  for (int i = 1; i < N + 1; i++) {
    for (int k = 1; k < N + 1; k++) {
      u[i][0][k] = 20; // z = 1
      u[i][N + 1][k] = 20; // z = -1
    }
  }

  for (int j = 1; j < N + 1; j++) {
    for (int k = 1; k < N + 1; k++) {
      u[0][j][k] = 0; // y = -1
      u[N + 1][j][k] = 20; // y = 1
    }
  }

  for (int i = 1; i < N + 1; i++) {
    for (int j = 1; j < N + 1; j++) {
      u[i][j][0] = 20; // x = -1
      u[i][j][N + 1] = 20; // x = 1
    }
  }
}

void initialize_data(double ***u, double ***f, int N) {
#pragma omp for schedule(static)
  for (int i = 0; i < N + 2; i++) {
    for (int j = 0; j < N + 2; j++) {
      for (int k = 0; k < N + 2; k++) {
        u[i][j][k] = 0.0;
        f[i][j][k] = 0.0;
      }
    }
  }

#pragma omp single
  {
    initialize_border(u, N);

    int i_min = 0; //y
    int j_min = (int) ((N + 2.0) / 2.0); // z
    int k_min = 0; // x
    int i_max = (int) ((N + 2.0) / 4.0); // y;
    int j_max = (int) ((N + 2.0) * 5.0 / 6.0); // z
    int k_max = (int) ((N + 2.0) * 1.5 / 8.0); // x

    for (int i = i_min; i < i_max; i++) {
      for (int j = j_min; j < j_max; j++) {
        for (int k = k_min; k < k_max; k++) {
          f[i][j][k] = 200;
        }
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
  char *output_prefix = "poisson";
#ifdef _JACOBI
  char *type = "j";
#endif
#ifdef _GAUSS_SEIDEL
  char *type = "gs";
#endif
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

  double allocation_t = 0;
  double initialize_t = 0;
  double compute_t = 0;

  double ***u = NULL;
  double ***u2 = NULL;
  double ***f = NULL;

  allocation_t -= omp_get_wtime();
  // allocate memory
  if ((u = malloc_3d(N + 2, N + 2, N + 2)) == NULL) {
    perror("array u: allocation failed");
    exit(-1);
  }

#ifdef _JACOBI
  if ((u2 = malloc_3d(N + 2, N + 2, N + 2)) == NULL) {
    perror("array u2: allocation failed");
    exit(-1);
  }
#endif

  if ((f = malloc_3d(N + 2, N + 2, N + 2)) == NULL) {
    perror("array f: allocation failed");
    exit(-1);
  }

  allocation_t += omp_get_wtime();

  initialize_t -= omp_get_wtime();

#pragma omp parallel
  {
    initialize_data(u, f, N);
#ifdef _JACOBI
    initialize_border(u2, N);
#endif
#pragma omp single
    {
      initialize_t += omp_get_wtime();

      compute_t -= omp_get_wtime();
    }
#ifdef _JACOBI
    solve_jacobi(u2, u, f, N, iter_max, tolerance);
#endif
#ifdef _GAUSS_SEIDEL
    solve_gauss_seidel(u, f, N, iter_max);
#endif
  }
  compute_t += omp_get_wtime();

  printf("%s, %f, %f, %f, %ld, %d\n",
         type,
         allocation_t,
         initialize_t,
         compute_t,
         (long) (iter_max) * N * N * N,
         iter_max);

  // dump  results if wanted
  switch (output_type) {
    case 0:
      // no output at all
      break;
    case 3: output_ext = ".bin";
      sprintf(output_filename, "%s_%s_%d_%d_%f_%s", output_prefix, type,
              N, iter_max, tolerance, output_ext);
      fprintf(stderr, "Write binary dump to %s: ", output_filename);
      print_binary(output_filename, N + 2, u);
      break;
    case 4: output_ext = ".vtk";
      sprintf(output_filename, "%s_%s_%d_%d_%f_%s", output_prefix, type,
              N, iter_max, tolerance, output_ext);
      fprintf(stderr, "Write VTK file to %s: ", output_filename);
      print_vtk(output_filename, N + 2, u);
      break;
    default: fprintf(stderr, "Non-supported output type!\n");
      break;
  }

  // de-allocate memory
  free_3d(u);
#ifdef _JACOBI
  free_3d(u2);
#endif
  free_3d(f);

  return (0);
}
