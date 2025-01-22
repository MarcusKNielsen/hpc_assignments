/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

void jacobi(double ***U_new_d0, double ***U_old_d0, double ***F_d0, double ***U_new_d1, double ***U_old_d1, double ***F_d1, int N);
int solve_jacobi(double ***U_new_d0, double ***U_old_d0, double ***F_d0, double ***U_new_d1, double ***U_old_d1, double ***F_d1, int N, int max_it, double threshold);

#endif
