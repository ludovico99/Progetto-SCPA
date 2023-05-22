#ifndef HEADER_H
#define HEADER_H

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

#ifdef CORRECTNESS
#define AUDIT if (1)
#else
#define AUDIT if (0)
#endif

#define BILLION 1000000000L

#ifdef SAMPLINGS
#define SAMPLING_SIZE 30
#ifdef CSR
extern void computing_samplings_openMP(int, int, int *, int, double *, int *, int *, int);
#elif ELLPACK
extern void computing_samplings_openMP(int, int, int *, int, int *, double **, int **, int);
#endif
#endif

#define SYMM_PATTERN 0
#define SYMM_REAL 1
#define GENERAL_PATTERN 2
#define GENERAL_REAL 3

#ifdef CSR
// CONVERSION CSR
extern void coo_to_CSR_serial(int, int, int, int *, int *, double *, double **, int **, int **);
extern int *coo_to_CSR_parallel(int, int, int, int *, int *, double *, double **, int **, int **, int);
extern int *coo_to_CSR_parallel_optimization(int, int, int, int *, int *, double *, double **, int **, int **, int);

void free_CSR_data_structures(double *, int *, int *);
#endif

#ifdef ELLPACK
// CONVERSION ELLPACK
extern int coo_to_ellpack_serial(int, int, int, int *, int *, double *, double ***, int ***);
extern int coo_to_ellpack_parallel(int, int, int, int *, int *, double *, double ***, int ***, int);
extern int *coo_to_ellpack_no_zero_padding_parallel(int, int, int, int *, int *, double *, double ***, int ***, int);
extern int *coo_to_ellpack_no_zero_padding_parallel_optimization(int, int, int, int *, int *, double *, double ***, int ***, int);

extern void free_ELLPACK_data_structures(int, double **, int **);
#endif

extern double **serial_product_CSR(int, int, int, int, double *, int *, int *, double **, double *);
extern double **serial_product_ellpack(int, int, int, int,int,  double **, int **, double **, double *);
extern double **serial_product_ellpack_no_zero_padding(int, int, int,int,  int *, double **, int **, double **, double *);

#ifdef OPENMP
// OPENMP CSR
extern double **parallel_product_CSR(int, int, int, int, double *, int *, int *, double **, double *, int);
// OPENMP ELLPACK
extern double **parallel_product_ellpack(int, int, int, int, int, double **, int **, double **, double *, int);
extern double **parallel_product_ellpack_no_zero_padding(int, int, int, int, int *, double **, int **, double **, double *, int);

extern void free_X(int, double **);

#endif

#ifdef CUDA

extern double *convert_2D_to_1D(int, int, double **);
extern double *convert_2D_to_1D_per_ragged_matrix(int, int, int *, double **);
extern int *convert_2D_to_1D_per_ragged_matrix(int, int, int *, int **);

#ifdef CSR
extern double *CSR_GPU(int, int, int, int, double *, int *, int *, double **, double *);
#elif ELLPACK
extern double *ELLPACK_GPU(int, int, int, int, int *, double **, int **, double **, double *);
#endif
#endif

extern double compute_GFLOPS(int, int, double);
extern int compute_chunk_size(int, int);
extern int *compute_sum_nz(int, int *);
extern void coo_general(int, int *, int *, int *, int **, int **, double **, int);
extern void coo_symm(int, int *, int *, int *, int **, int **, double **, int);
extern void create_dense_matrix(int, int, double ***);
extern void free_y(int, double **);
extern void print_y (int M, int K, double ** y);
extern void print_y_GPU (int M, int K, double *y);

#ifdef CHECK_CONVERSION
#ifdef CSR
extern int compare_conversion_algorithms_csr(int, int, int, int *, int *, double *, int);
#endif
#ifdef ELLPACK
extern int compare_conversion_algorithms_ellpack(int, int, int, int *, int *, double *, int);
#endif
#endif

extern char *filename;
extern FILE *f;

#endif
