02614: Assignment 1 - Specifications and tools
----------------------------------------------

This is an overview over the different specifications and tools for
the 1. assignment in the DTU course 02614.

Contents:

1. Coding style
2. Makefile templates
3. Drivers


1. Coding style
---------------

There are two different ways to implement the matrices: 

a) a native C/C++ way with double pointers, i.e. A(i,j) => A[i][j], or 

b) a representation where the matrix is implemented as a long vector, i.e.  
   A(i,j) => A[ ...i + ...j].  

If you choose version a), your function prototypes will look like

void matmult_nat(int m,int n,int k,double **A,double **B,double **C);
void matmult_blk(int m,int n,int k,double **A,double **B,double **C, int bs);

and in case of implentation b), correspondingly

void matmult_nat(int m,int n,int k,double *A,double *B,double *C);
void matmult_blk(int m,int n,int k,double *A,double *B,double *C, int bs);

All matmult_NNN() functions take 6 arguments, as show above, except
the blocked version, which will take an extra argument bs - the
blocksize. 

The two different implementations need two different drivers, and they
are denoted as _c for version a) and _f for version b) in the driver
programs (see below).


2. Makefile templates
---------------------

Different compilers need different options, and we provide 2 different
Makefile templates:

Makefile.gcc    - for C code compiled with gcc
Makefile.g++    - for C++ code compiled with g++

You'll need to modify the Makefile of your choice, i.e. adding the
list of source/object files needed to build the shared library, change
compiler options, etc.


3. Drivers
----------

The different programming styles (see 1. above) need different
drivers, and there is a dependency on the compiler as well.  The
drivers provided are:

a) matrices represented by double pointers (see 1.a):

matmult_c.gcc     - driver for libraries built with gcc/g++, linked
                    with CBLAS from ATLAS

b) matrices represented by single pointers, i.e. as a vector (see 1.b):

matmult_f.gcc     - driver for libraries built with gcc/g++, linked
                    with CBLAS from ATLAS

All drivers take the same command line arguments:

matmult_... type m k n [bs]

where m, n, k are the parameters defining the matrix sizes, bs is the
optional blocksize for the block version, and type can be one of:

nat	- the native/naive version
lib	- the library version (see above, which library will be called)
blk	- the blocked version (takes bs as extra argument)

as well as mnk, nmk, ... (the permutations).

To get reasonable results, even for small values of m, n and k, the
driver will run all functions for at least a minimum interval (more
details how to control this see the section on environment variables
below).

The output of a run looks like:

$ ./matmult_f.gcc nat 50 60 70
    83.594    151.173 0 # matmult_nat

where the first number is the memory usage in kbytes, the second
number the Mflop/s and the third number the difference between your
function and the reference implementation.  This number should be 0 or
at least very small - everything else indicates a problem in your
code.  The fourth field prints the name of the called libray function.

With the help of this driver program, you should be able to run all
experiments needed.

Note:  you need to build your own version of libmatmult.so first, before
you can use the driver!  There is no need to provide all the functions
at once, i.e. you can start with e.g. matmult_nat() and add the other
functions successively as you develop them.


Environment variables to control the driver:

There are 4 variables that control the behaviour of the driver
program.  They are

MATMULT_RESULTS={[0]|1}	  - print result matrices (in Matlab format, def: 0)
MATMULT_COMPARE={0|[1]}   - control result comparison (def: 1)
MFLOPS_MIN_T=[3.0]        - the minimum run-time (def: 3.0 s)
MFLOPS_MAX_IT=[infinity]  - max. no of iterations; 
                            set if you want to do profiling.

