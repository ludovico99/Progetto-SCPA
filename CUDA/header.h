#ifndef HEADER_H
#define HEADER_H


#ifdef CORRECTNESS
    #define AUDIT if (1)
#else 
    #define AUDIT if (0)
#endif


//SERIAL
extern void coo_to_CSR_serial(int , int , int , int *, int *, double *, double **, int **, int **);

extern int coo_to_ellpack_serial(int, int , int , int *, int *, double *, double ***, int ***);


//PARALLEL

extern void coo_to_CSR_parallel(int , int , int , int *, int *, double *, double **, int **, int **);
extern void coo_to_CSR_parallel_optimization(int , int , int , int *, int *, double *, double **, int **, int **);

extern int coo_to_ellpack_parallel(int , int , int , int *, int *, double *, double ***, int ***);
extern int* coo_to_ellpack_no_zero_padding_parallel(int , int , int , int *, int *, double *, double ***, int ***);
extern int *coo_to_ellpack_no_zero_padding_parallel_optimization(int , int , int , int *, int *, double *, double ***, int ***);


#endif