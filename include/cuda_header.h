#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

#ifdef SAMPLINGS
#define SAMPLING_SIZE 3
#endif

#define memory_allocation_Cuda(tipo, dimensione, puntatore)                                              \
    err = cudaMalloc((void **)&puntatore, dimensione * sizeof(tipo));                                    \
    if (err != cudaSuccess)                                                                              \
    {                                                                                                    \
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);                                                                                         \
    }

#define memcpy_to_dev(source, destination, tipo, dimensione)                                                  \
    err = cudaMemcpy(destination, source, dimensione * sizeof(tipo), cudaMemcpyHostToDevice);                 \
    if (err != cudaSuccess)                                                                                   \
    {                                                                                                         \
        fprintf(stderr, "Failed to copy as from host to device (error code %s)!\n", cudaGetErrorString(err)); \
        exit(1);                                                                                              \
    }

#define memcpy_to_host(source, destination, tipo, dimensione)                                                 \
    err = cudaMemcpy(destination, source, dimensione * sizeof(tipo), cudaMemcpyDeviceToHost);                 \
    if (err != cudaSuccess)                                                                                   \
    {                                                                                                         \
        fprintf(stderr, "Failed to copy as from device to host (error code %s)!\n", cudaGetErrorString(err)); \
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
extern double *ELLPACK_GPU(int, int, int, int, int *, double **, int **, double **, double *);

#elif CSR
extern double *CSR_GPU(int, int, int, int, double *, int *, int *, double **, double *);

#endif
#endif