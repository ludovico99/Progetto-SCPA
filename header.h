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


#ifdef CSR
    #include "csr_header.h"
#elif ELLPACK
    #include "ellpack_header.h"
#endif

#define BILLION 1000000000L


#define SYMM_PATTERN 0
#define SYMM_REAL 1
#define GENERAL_PATTERN 2
#define GENERAL_REAL 3

extern double compute_GFLOPS(int, int, double);
extern int compute_chunk_size(int, int);
extern int *compute_sum_nz(int, int *);
extern void coo_general(int, int *, int *, int *, int **, int **, double **, int);
extern void coo_symm(int, int *, int *, int *, int **, int **, double **, int);
extern void create_dense_matrix(int, int, double ***);
extern void free_y(int, double **);
extern void print_y (int, int, double **);
extern void print_y_GPU (int, int, double *);
extern double* transpose(int, int, double **);


extern char *filename;
extern FILE *f;

#endif
