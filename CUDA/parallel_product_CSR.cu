#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include <cuda_runtime.h> // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h> // For CUDA SDK timers
#include "../include/header.h"

/**
 * CSR_kernel_v1 -  Product implementation between sparse matrix A and dense matrix X
 *
 *@param M: Number of rows of the matrix A
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param X: Dense matrix
 *@param d_y: Resulting matrix
 *@param numElements: Number of elements of the product matrix Y
 *
 * Ogni thread ha il compito di computare un singolo elemento della matrice finale Y.
 * La riga dell'elemento viene computata tramite 'thread_id / K' mentre la colonna
 * tramite 'thread_id % K'. In questa versione del prodotto la somma parziale non è
 * ottimizzata poiché abbiamo un accesso alla memoria globale ogni volta che modifichiamo
 * il risultato intermedio.
 */
__global__ void CSR_kernel_v1(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int numElements)
{
    /* Thread identifier */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Row of the item that the thread should compute */
    int i = tid / K;

    /* Item column that the thread should compute */
    int z = tid % K;

    if (tid < numElements)
    {
        if (i == 0 && d_irp[i] == -1)
        {
            d_y[i * K + z] = 0.0;
        }
        if (i > 0 && d_irp[i] == d_irp[i - 1])
        {
            d_y[i * K + z] = 0.0;
        }
        else
        {
            for (int j = d_irp[i]; (i < (M - 1) && j < d_irp[i + 1]) || (i >= M - 1 && j < nz); j++)
            {
                if (d_as != NULL)
                    d_y[i * K + z] += d_as[j] * d_X[d_ja[j] * K + z];
                else
                    d_y[i * K + z] += 1.0 * d_X[d_ja[j] * K + z];
            }
        }
    }
}

/**
 * CSR_kernel_v2 - Product implementation between sparse matrix A and dense matrix X
 *
 *@param M: Number of rows of the matrix A
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param X: Dense matrix
 *@param d_y: Resulting matrix
 *@param numElements: Number of elements of the product matrix Y
 *
 * Ogni thread ha il compito di computare un singolo elemento della matrice finale Y.
 * La riga dell'elemento viene computata tramite 'thread_id / K' mentre la colonna
 * tramite 'thread_id % K'. In questa versione del prodotto la somma parziale è
 * ottimizzata poiché evitiamo di accedere continuamente alla memoria globale durante
 * il calcolo del valore dell'elemento che il thread deve computare.
 */
__global__ void CSR_kernel_v2(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int numElements)
{
    /* Thread identifier */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Row of the item that the thread should compute */
    int i = tid / K;

    /* Item column that the thread should compute */
    int z = tid % K;

    /* Partial result of matrix element Y */
    double partial_sum = 0;

    if (tid < numElements)
    {

        if (i == 0 && d_irp[i] == -1)
        {
            d_y[i * K + z] = 0.0;
        }
        else if (i > 0 && d_irp[i] == d_irp[i - 1])
        {
            d_y[i * K + z] = 0.0;
        }
        else
        {

            for (int j = d_irp[i]; (i < (M - 1) && j < d_irp[i + 1]) || (i >= M - 1 && j < nz); j++)
            {
                if (d_as != NULL)
                    partial_sum += d_as[j] * d_X[d_ja[j] * K + z];
                else
                    partial_sum += 1.0 * d_X[d_ja[j] * K + z];
            }
            d_y[i * K + z] = partial_sum;
        }
    }
}

/**
 * CSR_kernel_v3 - Product implementation between sparse matrix A and dense matrix X
 *
 *@param M: Number of rows of the matrix A
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param X: Dense matrix
 *@param d_y: Resulting matrix
 *@param numElements: Number of elements of the product matrix Y
 *
 * Ogni thread ha il compito di computare un singolo elemento della matrice finale Y.
 * La riga dell'elemento viene computata tramite 'thread_id / K' mentre la colonna
 * tramite 'thread_id % K'. In questa versione del prodotto si vuole ottimizzare il numero di accessi a d_irp
 * andando a memorizzarne il valore in una variabile automatica.
 */
__global__ void CSR_kernel_v3(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int numElements)
{
    /* Thread identifier */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Row of the item that the thread should compute */
    int i = tid / K;

    /* Item column that the thread should compute */
    int z = tid % K;

    /* Partial result of matrix element Y */
    double partial_sum = 0;

    if (tid < numElements)
    {
        int start = d_irp[i];
        int end = d_irp[i + 1];

        if (i == 0 && start == -1)
        {
            d_y[i * K + z] = 0.0;
        }
        else if (i > 0 && start == d_irp[i - 1])
        {
            d_y[i * K + z] = 0.0;
        }
        else
        {

            for (int j = start; (i < (M - 1) && j < end) || (i >= M - 1 && j < nz); j++)
            {
                if (d_as != NULL)
                    partial_sum += d_as[j] * d_X[d_ja[j] * K + z];
                else
                    partial_sum += 1.0 * d_X[d_ja[j] * K + z];
            }
            d_y[i * K + z] = partial_sum;
        }
    }
}

/**
 * CSR_Vector_Kernel - Product implementation between sparse matrix A and dense matrix X
 *
 *@param M: Number of rows of the matrix A
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param X: Dense matrix
 *@param d_y: Resulting matrix
 *@param numElements: Number of elements of the product matrix Y
 *
 *
 */
__global__ void CSR_Vector_Kernel(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int numElements)
{
    __shared__ double vals[1024];

    /* Thread identifier */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Global Warp Index */
    int tid_warp = tid / 32;

    /* TID Within the warp */
    int tid_within_warp = tid & (31);

    /* Row of the item that the warp should compute */
    int i = tid_warp / K;

    /* Column of the item that the warp should compute */
    // int z = tid_warp % K;

    if (i < M)
    {
        int start = d_irp[i];
        int end = d_irp[i + 1];

        vals[threadIdx.x] = 0.0;

        for (int z = 0; z < K; z++)
        {
            for (int j = start + tid_within_warp; (i < (M - 1) && j < end) || (i >= M - 1 && j < nz); j += 32)
            {
                if (d_as != NULL)
                    vals[threadIdx.x] += d_as[j] * d_X[d_ja[j] * K + z];
                else
                    vals[threadIdx.x] += 1.0 * d_X[d_ja[j] * K + z];
            }

            if (tid_within_warp < 16)
                vals[threadIdx.x] += vals[threadIdx.x + 16];

            if (tid_within_warp < 8)
                vals[threadIdx.x] += vals[threadIdx.x + 8];

            if (tid_within_warp < 4)
                vals[threadIdx.x] += vals[threadIdx.x + 4];

            if (tid_within_warp < 2)
                vals[threadIdx.x] += vals[threadIdx.x + 2];

            if (tid_within_warp < 1)
                vals[threadIdx.x] += vals[threadIdx.x + 1];

            if (tid_within_warp == 0)
                d_y[i * K + z] += vals[threadIdx.x];
        }
    }
}

/**
 *
 * CSR_GPU - This function performs setups to launch the kernel:
 *
 *   1. Dense matrix X is converted from 2D to 1D.
 *
 *   2. Memory allocation is performed for the Y matrix.
 *
 *   3. Memory is allocated to Device.
 *
 *   4. The contents of data structures are copied from the Host to the Device.
 *
 *   5. The kernel is launched.
 *
 *   6. The result is acquired from device
 *
 *   7. Memory allocated on both devices and hosts is released
 *
 *@param M: Number of rows of the matrix A
 *@param N: Number of columns of the matrix A, Number of rows of the matrix X
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param h_as: Vector containing the non-zero elements of the sparse array
 *@param h_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param h_irp: Vector containing the column index of the first nonzero of rows
 *@param X: Dense matrix
 *@param time: Pointer to a double representing the time elapsed for the GPU product
 *
 * Returns the resulting/product matrix computed by the GPU kernel
 */

double *CSR_GPU(int M, int N, int K, int nz, double *h_as, int *h_ja, int *h_irp, double **X, double *time)
{
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;

    // HOST
    double *h_y = NULL;
    double *h_X = NULL;

    // DEVICE
    double *d_y = NULL;
    double *d_X = NULL;
    double *d_as = NULL;
    int *d_ja = NULL;
    int *d_irp = NULL;

    float expireTimeMsec = 0.0;

    /* 2D to 1D dense matrix X conversion*/
    h_X = convert_2D_to_1D(M, K, X);

    /* Y array host memory allocation */
    memory_allocation(double, M *K, h_y);

    printf("Allocating device variables for CPU CSR product ...\n");

    /* Y array host memory allocation */
    memory_allocation_Cuda(double, M *K, d_y);
    /* Device allocation for dense matrix X */
    memory_allocation_Cuda(double, N *K, d_X);
    /* Device allocation for the as vector containing non-zero elements */
    memory_allocation_Cuda(double, nz, d_as);
    /* Device allocation for the as vector containing non-zero elements */
    memory_allocation_Cuda(int, nz, d_ja);
    /* Device allocation for the irp vector containing the pointer to the vector entry ja */
    memory_allocation_Cuda(int, M, d_irp);

    printf("Copy input data from the host memory to the CUDA device\n");

    /* Copy of the contents of the vector as from the Host to the Device */
    memcpy_to_dev(h_as, d_as, double, nz);
    /* Copy of the contents of the vector ja from the Host to the Device */
    memcpy_to_dev(h_ja, d_ja, int, nz);
    /* Copy of the contents of the vector irp from the Host to the Devicee */
    memcpy_to_dev(h_irp, d_irp, int, M);
    /* Copy of the dense vector X from the Host to the Device*/
    memcpy_to_dev(h_X, d_X, double, N *K);

    /* Number of elements of the product matrix Y */
    int numElements = M * K;

    /* Number of threads per block */
    int threadsPerBlock = 1024;

    int warpPerBlock = threadsPerBlock / 32;

    /* Number of blocks per grid */
    int blocksPerGrid = (M + warpPerBlock - 1) / warpPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
           threadsPerBlock);

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // START TIMER
    checkCudaErrors(cudaEventRecord(start, stream));

    /* Versione accesso alla memoria globale non ottimizzato */
    // CSR_kernel_v1<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, numElements);

    /* Versione accesso alla memoria globale ottimizzato */
    // CSR_kernel_v3<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, numElements);

    /* CSR Vector */
    CSR_Vector_Kernel<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, numElements);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CSR kernel (error code %s)!\n",
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

    printf("Copy output data from the CUDA device to the host memory\n");

    /* Copy of the product matrix Y from the Device to the Host */
    memcpy_to_host(d_y, h_y, double, M *K);

    /* Start the memory cleaning process on Device */
    printf("Freeing Device memory ...\n");

    free_memory_Cuda(d_as);
    free_memory_Cuda(d_ja);
    free_memory_Cuda(d_irp);
    free_memory_Cuda(d_X);
    free_memory_Cuda(d_y);

    /* Start the memory cleaning process on Host */

    printf("Freeing host memory ...\n");

    free(h_X);

    printf("Completed parallel product CSR without streams...\n");

    return h_y;
}
