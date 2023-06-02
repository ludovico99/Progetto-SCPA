#ifndef CSR_HEADER_H
#define CSR_HEADER_H

#ifdef SAMPLINGS
extern void computing_samplings_openMP(int, int, int *, int, double *, int *, int *, int);
#endif

#ifdef CHECK_CONVERSION
extern int compare_conversion_algorithms_csr(int, int, int, int *, int *, double *, int);
#endif

// CONVERSION CSR
extern int *coo_to_CSR_serial(int, int, int, int *, int *, double *, double **, int **, int **);
extern int *coo_to_CSR_parallel(int, int, int, int *, int *, double *, double **, int **, int **, int);
extern int *coo_to_CSR_parallel_optimization(int, int, int, int *, int *, double *, double **, int **, int **, int);

extern double **serial_product_CSR(int, int, int, int, double *, int *, int *, double **, double *);


#endif