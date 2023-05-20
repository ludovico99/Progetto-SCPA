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

#define SYMM_PATTERN 0
#define SYMM_REAL 1
#define GENERAL_PATTERN 2
#define GENERAL_REAL 3

#ifdef CSR
// CONVERSION CSR
extern void coo_to_CSR_serial(int, int, int, int *, int *, double *, double **, int **, int **);
extern int * coo_to_CSR_parallel(int, int, int, int *, int *, double *, double **, int **, int **);
extern int * coo_to_CSR_parallel_optimization(int , int , int , int *, int *, double *, double **, int **, int **);
#endif

#ifdef ELLPACK
// CONVERSION ELLPACK
extern int coo_to_ellpack_serial(int, int, int, int *, int *, double *, double ***, int ***);
extern int coo_to_ellpack_parallel(int, int, int, int *, int *, double *, double ***, int ***);
extern int *coo_to_ellpack_no_zero_padding_parallel(int, int, int, int *, int *, double *, double ***, int ***);
extern int *coo_to_ellpack_no_zero_padding_parallel_optimization(int, int, int, int *, int *, double *, double ***, int ***);
#endif

extern double **serial_product_CSR(int, int, int, int, double *, int *, int *, double **, double *);

#ifdef OPENMP
// OPENMP CSR
extern double **parallel_product_CSR(int , int , int , int , double *, int *, int *, double **,double * , int );
// OPENMP ELLPACK
extern double **serial_product_ellpack(int, int, int, int, double **, int **, double **, double *);
extern double **serial_product_ellpack_no_zero_padding(int, int, int, int *, double **, int **, double **, double *);
extern double **parallel_product_ellpack(int, int, int, int, double **, int **, double **, double *, int);
extern double **parallel_product_ellpack_no_zero_padding(int, int, int, int *, double **, int **, double **, double *, int);
#endif

#ifdef CUDA
extern double *CSR_GPU(int, int, int, int, double *, int *, int *, double **, double *);
#endif

extern double compute_GFLOPS (int k, int nz, double time);
extern int compute_chunk_size(int , int);
extern void coo_general(int , int *, int *, int *, int **, int **, double **, int );
extern void coo_symm(int , int *, int *, int *, int **, int **, double **, int );

#ifdef CHECK_CONVERSION
#ifdef CSR
extern int compare_conversion_algorithms_csr(int , int , int , int *, int *, double *);
#endif
#ifdef ELLPACK
extern int compare_conversion_algorithms_ellpack(int , int , int , int *, int *, double *);
#endif
#endif

extern char *filename;
extern FILE * f;

#endif
