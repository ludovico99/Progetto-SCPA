#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include <cuda_runtime.h> // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h> // For CUDA SDK timers

#include "../header.h"

__global__ void ELLPACK_kernel(const int M, const int K, int *nz_per_row, int * sum_nz, double *d_values, int *d_col_indices, double *d_X, double *d_y, int numElements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int i = tid / K;
    int z = tid % K;
    double partial_sum = 0.0;
    int offset = sum_nz[i];
    if (tid < numElements)
    {
        if (nz_per_row[i] == 0)
            d_y[i * K + z] = 0.0;
        else
        {
            for (int j = 0; j < nz_per_row[i]; j++)
            {
                if (d_values != NULL)
                    partial_sum += d_values[offset + j] * d_X[d_col_indices[offset + j] * K + z];
                else
                    partial_sum += 1.0 * d_X[d_col_indices[offset + j] * K + z];
            }
            d_y[i * K + z] = partial_sum;
        }
    }
}

double *ELLPACK_GPU(int M, int N, int K, int nz, int *nz_per_row, double **values, int **col_indices, double **X, double *time)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;

    double *h_y = NULL;
    double *d_y = NULL;

    double *h_X = NULL;
    double *d_X = NULL;

    double *h_values = NULL;
    int *h_col_indices = NULL;

    double *d_values = NULL;
    int *d_col_indices = NULL;
    int *d_nz_per_row = NULL;

    int * h_sum_nz = NULL;
    int * d_sum_nz = NULL;

    float expireTimeMsec = 0.0;

    h_X = convert_2D_to_1D(M, K, X);

    memory_allocation(double, M *K, h_y);
 
    h_values = convert_2D_to_1D_per_ragged_matrix(M, nz, nz_per_row, values);
    h_col_indices = convert_2D_to_1D_per_ragged_matrix(M, nz, nz_per_row, col_indices);

    printf("Allocating device variables for CPU ELLPACK product ...\n");
    /* Allocazione su Device per la matrice Y */
    memory_allocation_Cuda(double, M *K, d_y);
    /* Allocazione su Device per la matrice densa X */
    memory_allocation_Cuda(double, N * K, d_X);

    memory_allocation_Cuda(double, nz, d_values);

    memory_allocation_Cuda(int, nz, d_col_indices);

    memory_allocation_Cuda(int, M, d_nz_per_row);

    memory_allocation_Cuda(int, M, d_sum_nz);

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in device memory

    printf("Copy input data from the host memory to the CUDA device\n");

    memcpy_to_dev(h_values, d_values, double, nz);
    memcpy_to_dev(h_col_indices, d_col_indices, int, nz);
    memcpy_to_dev(h_X, d_X, double, N * K );
    memcpy_to_dev(nz_per_row, d_nz_per_row, int, M);
    
    h_sum_nz = compute_sum_nz(M, nz_per_row);
    memcpy_to_dev(h_sum_nz, d_sum_nz, int, M);

    // Launch the Vector Add CUDA Kernel
    int numElements = M * K;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
           threadsPerBlock);

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // START TIMER
    checkCudaErrors(cudaEventRecord(start, stream));

    ELLPACK_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, K, d_nz_per_row, d_sum_nz, d_values, d_col_indices, d_X, d_y, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch ELLPACK kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // STOP TIMER
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));

    printf("ELAPSED TIME FOR PARALLEL PRODUCT GPU: %lf ns = %lf ms = %lf seconds\n", expireTimeMsec * 1e6, expireTimeMsec, expireTimeMsec * 1e-3);

    if (time != NULL)
        *time = expireTimeMsec * 1e6;
    printf("GFLOPS FOR PARALLEL PRODUCT GPU: %lf\n", compute_GFLOPS(K, nz, expireTimeMsec * 1e6));

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");

     /* Copio la matrice prodotto Y dal Device all'Host */
    memcpy_to_host(d_y, h_y, double, M *K);

    printf("Freeing Device memory ...\n");
    // Free device global memory
    free_memory_Cuda(d_values);
    free_memory_Cuda(d_col_indices);
    free_memory_Cuda(d_nz_per_row);
    free_memory_Cuda(d_sum_nz);
    free_memory_Cuda(d_X);
    free_memory_Cuda(d_y);

    // Free host memory
    printf("Freeing host memory ...\n");
    if (h_X != NULL) free(h_X);
    if (h_values != NULL) free(h_values);
    if (h_col_indices != NULL) free(h_col_indices);
    if (h_sum_nz != NULL) free(h_sum_nz);

    printf("Completed parallel product ...\n");

    return h_y;
}