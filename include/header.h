#ifndef HEADER_H
#define HEADER_H

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <time.h>

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

#define memory_allocation(tipo, dimensione, puntatore) \
    puntatore = (tipo*)malloc(sizeof(tipo) * dimensione); \
    if (puntatore == NULL) { \
        fprintf(stderr, "Errore nell'allocazione di memoria con malloc.\n"); \
        perror("Errore Malloc: ");                                          \
        exit(1); \
    } \

#define all_zeroes_memory_allocation(tipo, dimensione, puntatore) \
    puntatore = (tipo*)calloc(dimensione , sizeof(tipo)); \
    if (puntatore == NULL) { \
        fprintf(stderr, "Errore nell'allocazione di memoria con calloc.\n"); \
        perror("Errore Malloc: ");  \
        exit(1); \
    } \


#ifdef CUDA

#define memory_allocation_Cuda(tipo, dimensione, puntatore) \
    err = cudaMalloc((void **)&puntatore, dimensione * sizeof(tipo)); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));\
        exit(1);\
    }\

    #define memcpy_to_dev(source, destination, tipo, dimensione) \
    err = cudaMemcpy(destination, source, dimensione * sizeof(tipo), cudaMemcpyHostToDevice);\
    if (err != cudaSuccess){ \
        fprintf(stderr, "Failed to copy as from host to device (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);\
    }\

    #define memcpy_to_host(source, destination, tipo, dimensione) \
    err = cudaMemcpy(destination, source, dimensione * sizeof(tipo), cudaMemcpyDeviceToHost);\
    if (err != cudaSuccess){ \
        fprintf(stderr, "Failed to copy as from device to host (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);\
    }\

    #define free_memory_Cuda(puntatore) \
    err = cudaFree(puntatore); \
    if (err != cudaSuccess) {   \
        fprintf(stderr, "Failed to free device memory (error code %s)!\n",  cudaGetErrorString(err));\
        exit(1);\
    }\
   
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
extern void get_time (struct timespec *);

extern char *filename;
extern FILE *f;

#endif
