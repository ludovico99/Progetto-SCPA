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
 *
 * ELLPACK_kernel - Product implementation between sparse matrix A and dense matrix X
 *
 *@param M: Number of rows of the matrix A
 *@param N: Number of columns of the matrix A, Number of rows of the matrix X
 *@param K:  Number of columns of the matrix X
 *@param nz_per_row: Array containing the number of non-zeroes per row
 *@param sum_nz: Array whose i-th entry contains the sum of non-zeroes up to the i-th row
 *@param d_values: 2D array of coefficients
 *@param d_col_indices: 2D array of column indexes
 *@param d_X: Dense matrix
 *@param d_y: Resulting matrix
 *@param numElements: Number of elements of the product matrix Y
 *
 */

__global__ void ELLPACK_kernel(const int M, const int K, int *nz_per_row, int *sum_nz, double *d_values, int *d_col_indices, double *d_X, double *d_y, int numElements)
{
    /* Thread identifier */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Row of the item that the thread should compute */
    int i = tid / K;

    /* Item column that the thread should compute */
    int z = tid % K;

    /* Partial result of matrix element Y */
    double partial_sum = 0;

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
/**
 *
 * ELLPACK_GPU - This function performs setups to launch the kernel:
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
 *@param nz_per_row: Array containing the number of non-zeroes per row
 *@param values: 2D array of coefficients
 *@param col_indices: 2D array of column indexes
 *@param X: Dense matrix
 *@param time: Pointer to a double representing the time elapsed for the GPU product
 *
 * Returns the resulting/product matrix computed by the GPU kernel
 */

double *ELLPACK_GPU(int M, int N, int K, int nz, int *nz_per_row, double **values, int **col_indices, double **X)
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

    int *h_sum_nz = NULL;
    int *d_sum_nz = NULL;

    float expireTimeMsec = 0.0;

    /* 2D to 1D dense matrix X conversion*/
    h_X = convert_2D_to_1D(N, K, X);

    /* Y array host memory allocation */
    memory_allocation(double, M *K, h_y);

    /* irregular 2D to 1D conversions*/
    if (values != NULL)
        h_values = convert_2D_to_1D_per_ragged_matrix(M, nz, nz_per_row, values);
    h_col_indices = convert_2D_to_1D_per_ragged_matrix(M, nz, nz_per_row, col_indices);

    printf("Allocating device variables for CPU ELLPACK product ...\n");
    /* Y array host memory allocation */
    memory_allocation_Cuda(double, M *K, d_y);
    /* Device allocation for dense matrix X */
    memory_allocation_Cuda(double, N *K, d_X);
    if (values != NULL)
        /* Device allocation for the 2D array ontaining non-zero elements */
        memory_allocation_Cuda(double, nz, d_values);
    /* Device allocation for the 2D array of column indexes */
    memory_allocation_Cuda(int, nz, d_col_indices);
    /* Device allocation for the array of non-zeroes per row*/
    memory_allocation_Cuda(int, M, d_nz_per_row);
    /* Device allocation for the array of the sums of non-zeroes*/
    memory_allocation_Cuda(int, M, d_sum_nz);

    printf("Copy input data from the host memory to the CUDA device\n");
    if (values != NULL)
        /* Copy of the contents of the h_values from the Host to the Device */
        memcpy_to_dev(h_values, d_values, double, nz);
    /* Copy of the contents of h_col_indices from the Host to the Device */
    memcpy_to_dev(h_col_indices, d_col_indices, int, nz);
    /* Copy of the contents of the dense vector h_X from the Host to the Devicee */
    memcpy_to_dev(h_X, d_X, double, N *K);
    /* Copy of the contents of nz_per_row from the Host to the Device */
    memcpy_to_dev(nz_per_row, d_nz_per_row, int, M);

    h_sum_nz = compute_sum_nz(M, nz_per_row);
    /* Copy of the contents of h_sum_nz from the Host to the Device */
    memcpy_to_dev(h_sum_nz, d_sum_nz, int, M);

    /* Number of elements of the product matrix Y */
    int numElements = M * K;
    /* Number of threads per block */
    int threadsPerBlock = 1024;
    /* Number of blocks per grid */
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

    printf("GFLOPS FOR PARALLEL PRODUCT GPU: %lf\n", compute_GFLOPS(K, nz, expireTimeMsec * 1e6));

    printf("Copy output data from the CUDA device to the host memory\n");

    /* Copy of the product matrix Y from the Device to the Host */
    memcpy_to_host(d_y, h_y, double, M *K);
    /* Start the memory cleaning process on Device */
    printf("Freeing Device memory ...\n");
    if (values != NULL)
        free_memory_Cuda(d_values);
    free_memory_Cuda(d_col_indices);
    free_memory_Cuda(d_nz_per_row);
    free_memory_Cuda(d_sum_nz);
    free_memory_Cuda(d_X);
    free_memory_Cuda(d_y);

    /* Start the memory cleaning process on Host */
    printf("Freeing host memory ...\n");
    if (h_X != NULL)
        free(h_X);
    if (h_values != NULL)
        free(h_values);
    if (h_col_indices != NULL)
        free(h_col_indices);
    if (h_sum_nz != NULL)
        free(h_sum_nz);

    printf("Completed parallel product ...\n");

    return h_y;
}