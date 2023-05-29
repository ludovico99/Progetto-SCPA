#ifndef OPENMP_HEADER_H
#define OPENMP_HEADER_H

extern void check_correctness(int, int, double **, double **);

#ifdef ELLPACK
extern double **parallel_product_ellpack(int, int, int, int, int, double **, int **, double **, double *, int);
extern double **parallel_product_ellpack_no_zero_padding(int, int, int, int, int *, double **, int **, double **, double *, int);
extern void free_ELLPACK_data_structures(int, double **, int **);
#elif CSR
extern double **parallel_product_CSR(int, int, int, int, double *, int *, int *, double **, double *, int);
extern void free_CSR_data_structures(double *, int *, int *);
#endif

extern void free_X(int, double **);
#endif