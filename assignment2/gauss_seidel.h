/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H

int solve_gauss_seidel(double ***u, double ***f, int N, int max_it, double threshold);
double gauss_seidel(double ***u, double ***f, int N);

void parallel_gauss_seidel(double ***u, double ***f, int N);
void parallel_solve_gauss_seidel(double ***u, double ***f, int N, int max_it);

#endif
