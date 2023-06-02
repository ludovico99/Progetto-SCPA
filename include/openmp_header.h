#ifndef OPENMP_HEADER_H
#define OPENMP_HEADER_H

#ifdef SAMPLINGS
#define SAMPLING_SIZE 30
#endif

extern void check_correctness(int, int, double **, double **);

#ifdef ELLPACK
extern double **parallel_product_ellpack(int, int, int, int, int, double **, int **, double **, double *, int);
extern double **parallel_product_ellpack_no_zero_padding(int, int, int, int, int *, double **, int **, double **, double *, int);

#elif CSR
extern double **parallel_product_CSR(int, int, int, int, double *, int *, int *, double **, double *, int);
#endif

extern void free_X(int, double **);
#endif