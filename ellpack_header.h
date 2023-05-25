#ifndef ELLPACK_HEADER_H
#define ELLPACK_HEADER_H

#ifdef SAMPLINGS
#define SAMPLING_SIZE 30
extern void computing_samplings_openMP(int, int, int *, int, int *, double **, int **, int);
#endif

#ifdef CHECK_CONVERSION
extern int compare_conversion_algorithms_ellpack(int, int, int, int *, int *, double *, int);
#endif

// CONVERSION ELLPACK
extern int coo_to_ellpack_serial(int, int, int, int *, int *, double *, double ***, int ***);
extern int coo_to_ellpack_parallel(int, int, int, int *, int *, double *, double ***, int ***, int);
extern int *coo_to_ellpack_no_zero_padding_parallel(int, int, int, int *, int *, double *, double ***, int ***, int);
extern int *coo_to_ellpack_no_zero_padding_parallel_optimization(int, int, int, int *, int *, double *, double ***, int ***, int);

extern double **serial_product_ellpack(int, int, int, int,int,  double **, int **, double **, double *);
extern double **serial_product_ellpack_no_zero_padding(int, int, int,int,  int *, double **, int **, double **, double *);


#ifdef OPENMP
// OPENMP ELLPACK
extern double **parallel_product_ellpack(int, int, int, int, int, double **, int **, double **, double *, int);
extern double **parallel_product_ellpack_no_zero_padding(int, int, int, int, int *, double **, int **, double **, double *, int);

extern void free_X(int, double **);
#endif


#ifdef CUDA

extern double *convert_2D_to_1D(int, int, double **);
extern double *convert_2D_to_1D_per_ragged_matrix(int, int, int *, double **);
extern int *convert_2D_to_1D_per_ragged_matrix(int, int, int *, int **);

extern double *ELLPACK_GPU(int, int, int, int, int *, double **, int **, double **, double *);
#endif

extern void free_ELLPACK_data_structures(int, double **, int **);

#endif