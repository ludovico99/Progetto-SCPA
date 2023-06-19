#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

#include <cuda_runtime.h> // For CUDA runtime API

#ifdef SAMPLINGS
#define SAMPLING_SIZE 10
#endif

#define MAX_BLOCK_DIM 512
#define WARP_SIZE 32

#define memory_allocation_Cuda(tipo, dimensione, puntatore)                                              \
    err = cudaMalloc((void **)&puntatore, (dimensione) * sizeof(tipo));                                    \
    if (err != cudaSuccess)                                                                              \
    {                                                                                                    \
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);                                                                                         \
    }

#define memcpy_to_dev(source, destination, tipo, dimensione)                                                  \
    err = cudaMemcpy(destination, source, (dimensione) * sizeof(tipo), cudaMemcpyHostToDevice);                 \
    if (err != cudaSuccess)                                                                                   \
    {                                                                                                         \
        fprintf(stderr, "Failed to copy data from host to device (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);                                                                                              \
    }

#define memcpy_to_host(source, destination, tipo, dimensione)                                                 \
    err = cudaMemcpy(destination, source, (dimensione) * sizeof(tipo), cudaMemcpyDeviceToHost);                 \
    if (err != cudaSuccess)                                                                                   \
    {                                                                                                         \
        fprintf(stderr, "Failed to copy data from device to host (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);                                                                                              \
    }

#define free_memory_Cuda(puntatore)                                                                  \
    err = cudaFree(puntatore);                                                                       \
    if (err != cudaSuccess)                                                                          \
    {                                                                                                \
        fprintf(stderr, "Failed to free device memory (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);                                                                                     \
    }

extern void check_correctness(int, int, double **, double *);

extern double *convert_2D_to_1D(int, int, double **);
extern double *convert_2D_to_1D_per_ragged_matrix(int, int, int *, double **);
extern int *convert_2D_to_1D_per_ragged_matrix(int, int, int *, int **);

#ifdef ELLPACK
extern double *ELLPACK_GPU(int, int, int, int, int *, double **, int **, double **);
extern void samplings_GPU_ELLPACK(int, int, int, int *, double **, int **);

extern __global__ void ELLPACK_kernel(const int, const int, int *, int *, double *, int *, double *, double *);
extern __global__ void ELLPACK_Sub_warp(const int , const int , int *, int *, double *, int *, double *, double *, const int);

#elif CSR
extern double *CSR_GPU(int, int, int, int, double *, int *, int *, double **);
extern void samplings_GPU_CSR(int, int, int, double *, int *, int *);

extern __global__ void CSR_kernel_v1(const int, const int, const int, double *, int *, int *, double *, double *);
extern __global__ void CSR_kernel_v2(const int, const int, const int, double *, int *, int *, double *, double *);
extern __global__ void CSR_kernel_v3(const int, const int, const int, double *, int *, int *, double *, double *);
extern __global__ void CSR_Vector_Sub_warp(const int, const int, const int, double *, int *, int *, double *, double *, const int);
extern __global__ void CSR_Vector_Kernel(const int, const int, const int, const int, double *, int *, int *, double *, double *);
extern __global__ void CSR_Adaptive_Kernel(const int, const int,  const int, const int, double *, int *, int *, double *, double *, int *);
extern int csr_adaptive_rowblocks(int, int, int *, int **, int *);

#endif
#endif