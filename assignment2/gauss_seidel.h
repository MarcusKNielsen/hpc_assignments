/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H


void solve_gauss_seidel(double ***u, double ***f, int N, int max_it);
void gauss_seidel(double ***u, double ***f, int N);

#endif
