#ifndef HEADER_H
#define HEADER_H

#define AUDIT if (0)
#define BILLION 1000000000L


//SERIAL
extern void coo_to_CSR_serial(int , int , int , int *, int *, double *, double **, int **, int **);
extern int coo_to_ellpack_serial(int, int , int , int *, int *, double *, double ***, int ***);

extern double **serial_product_CSR(int , int , int , int , double *, int *, int *, double **);
extern double **serial_product_ellpack(int , int , int , int , double **, int **, double **);
extern double **serial_product_ellpack_no_zero_padding(int, int, int , int* , double **, int **, double **);
extern double **serial_product(int , int , int , double **, double **);

//PARALLEL

extern void coo_to_CSR_parallel(int , int , int , int *, int *, double *, double **, int **, int **);
extern int coo_to_ellpack_parallel(int , int , int , int *, int *, double *, double ***, int ***);
extern int *coo_to_ellpack_no_zero_padding_parallel(int , int , int , int *, int *, double *, double ***, int ***);

extern double ** parallel_product_CSR(int, int, int, int, double *, int *, int *, double **, int );
extern double ** parallel_product_ellpack(int , int , int , int , double **, int **, double **);
extern double ** parallel_product_ellpack_no_zero_padding(int , int , int , int*, double **, int **, double **, *double);


#endif