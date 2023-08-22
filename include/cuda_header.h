#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

#include <cuda_runtime.h> // For CUDA runtime API

#ifdef SAMPLINGS
#define SAMPLING_SIZE 10
#endif

#define MAX_BLOCK_DIM 512
#define WARP_SIZE 32

#define adaptive 0
#define adaptive_sub_blocks 1

/* This is threshold for adaptive algorithm */
//#define THR 10

/* This is the size of a sub warp used in the adaptive algorithm */
#define SUB_WARP_SIZE 2

#define memory_allocation_Cuda(tipo, dimensione, puntatore)                                              \
    err = cudaMalloc((void **)&puntatore, (dimensione) * sizeof(tipo));                                  \
    if (err != cudaSuccess)                                                                              \
    {                                                                                                    \
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);                                                                                         \
    }

#define memcpy_to_dev(source, destination, tipo, dimensione)                                                    \
    err = cudaMemcpy(destination, source, (dimensione) * sizeof(tipo), cudaMemcpyHostToDevice);                 \
    if (err != cudaSuccess)                                                                                     \
    {                                                                                                           \
        fprintf(stderr, "Failed to copy data from host to device (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);                                                                                                \
    }

#define memcpy_to_host(source, destination, tipo, dimensione)                                                   \
    err = cudaMemcpy(destination, source, (dimensione) * sizeof(tipo), cudaMemcpyDeviceToHost);                 \
    if (err != cudaSuccess)                                                                                     \
    {                                                                                                           \
        fprintf(stderr, "Failed to copy data from device to host (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);                                                                                                \
    }

#define free_memory_Cuda(puntatore)                                                                  \
    err = cudaFree(puntatore);                                                                       \
    if (err != cudaSuccess)                                                                          \
    {                                                                                                \
        fprintf(stderr, "Failed to free device memory (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);                                                                                     \
    }

struct item {
    int row;          //ROW
    int col;          //COLUMN
};

struct core_adaptive_personalizzato
{
    long *metadata;
    //Per ogni thread mantiene l'elemento di Y che dovrà calcolare
    struct item* items_scalar;
    //Per ogni thread mantiene l'elemento di Y che dovrà calcolare
    struct item* items_vector;   
};

extern void check_correctness(int, int, double **, double *);

extern double *convert_2D_to_1D(int, int, double **);
extern double *convert_2D_to_1D_per_ragged_matrix(int, int, int *, double **);
extern int *convert_2D_to_1D_per_ragged_matrix(int, int, int *, int **);

#ifdef ELLPACK
extern double *ELLPACK_GPU(int, int, int, int, int *, double **, int **, double **);
extern void samplings_GPU_ELLPACK(int, int, int, int *, double **, int **);

extern __global__ void ELLPACK_kernel(const int, const int, int *, int *, double *, int *, double *, double *);
extern __global__ void ELLPACK_Sub_warp(const int, const int, int *, int *, double *, int *, double *, double *, const int);

#elif CSR
extern double *CSR_GPU(int, int, int, int, double *, int *, int *, double **, int *);
extern void samplings_GPU_CSR(int, int, int, double *, int *, int *, int *);
extern void samplings_GPU_CSR_flush_cache(int, int, int, double *, int *, int *, int *);

extern __global__ void CSR_Scalar_v1(const int, const int, const int, double *, int *, int *, double *, double *);
extern __global__ void CSR_Scalar_v2(const int, const int, const int, double *, int *, int *, double *, double *);
extern __global__ void CSR_Scalar_v3(const int, const int, const int, double *, int *, int *, double *, double *);
extern __global__ void CSR_Vector_Sub_warp(const int, const int, const int, double *, int *, int *, double *, double *, const int);
extern __global__ void CSR_Vector(const int, const int, const int, const int, double *, int *, int *, double *, double *);
extern __global__ void CSR_Vector_by_row(const int, const int, const int, const int, double *, int *, int *, double *, double *);
extern __global__ void CSR_Adaptive(const int, const int, const int, const int, double *, int *, int *, double *, double *, int *);
extern __global__ void CSR_Adaptive_personalizzato(const int, const int, const int, const int, double *, int *, int *, double *, double *, long *, struct item*, struct item*);
extern int csr_adaptive_rowblocks(int, int, int, int *, int **, int *);
extern struct core_adaptive_personalizzato *csr_adaptive_personalizzato_number_of_blocks(int, int *, int, int);

#endif
#endif
