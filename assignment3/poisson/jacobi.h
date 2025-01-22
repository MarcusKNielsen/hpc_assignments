/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

int solve_jacobi(double ***U_new, double ***U_old, double ***F, int N, int max_it, double threshold);
void jacobi(double ***U_new, double ***U_old, double ***F, int N);

#endif
