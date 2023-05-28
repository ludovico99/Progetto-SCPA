#ifndef CSR_HEADER_H
#define CSR_HEADER_H

#ifdef SAMPLINGS
#define SAMPLING_SIZE 30
extern void computing_samplings_openMP(int, int, int *, int, double *, int *, int *, int);
#endif

#ifdef CHECK_CONVERSION
extern int compare_conversion_algorithms_csr(int, int, int, int *, int *, double *, int);
#endif

// CONVERSION CSR
extern void coo_to_CSR_serial(int, int, int, int *, int *, double *, double **, int **, int **);
extern int *coo_to_CSR_parallel(int, int, int, int *, int *, double *, double **, int **, int **, int);
extern int *coo_to_CSR_parallel_optimization(int, int, int, int *, int *, double *, double **, int **, int **, int);

extern double **serial_product_CSR(int, int, int, int, double *, int *, int *, double **, double *);

#ifdef OPENMP
// OPENMP CSR
extern double **parallel_product_CSR(int, int, int, int, double *, int *, int *, double **, double *, int);

extern void free_X(int, double **);

#endif

#ifdef CUDA

extern double *convert_2D_to_1D(int, int, double **);
extern double *convert_2D_to_1D_per_ragged_matrix(int, int, int *, double **);
extern int *convert_2D_to_1D_per_ragged_matrix(int, int, int *, int **);

extern double *CSR_GPU(int, int, int, int, double *, int *, int *, double **, double *);
#endif

void free_CSR_data_structures(double *, int *, int *);

#endif