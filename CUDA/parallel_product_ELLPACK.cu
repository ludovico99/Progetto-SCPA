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
 * Each threads computes an element in the resulting matrix. 
 * The row and the columns assigned to each thread depends on the global tid.
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

__global__ void ELLPACK_kernel(const int M, const int K, int *nz_per_row, int *sum_nz, double *d_values, int *d_col_indices, double *d_X, double *d_y)
{
    const int num_elements = M * K;

    /* Thread identifier */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Row of the item that the thread should compute */
    const int i = tid / K;

    /* Item column that the thread should compute */
    const int z = tid % K;

    if (tid < num_elements)
    {   
        const int offset = sum_nz[i];

        const int max_nz = nz_per_row[i];

        /* Partial result of matrix element Y */
        double partial_sum = 0.0;
        for (int j = 0; j < max_nz; j++)
        {
            if (d_values != NULL)
                partial_sum += d_values[offset + j] * d_X[d_col_indices[offset + j] * K + z];
            else
                partial_sum += 1.0 * d_X[d_col_indices[offset + j] * K + z];
        }
        d_y[i * K + z] = partial_sum;
    }
}

/**
 * ELLPACK_Sub_warp - Product implementation between sparse matrix A and dense matrix X.
 * 
 * Each thread in a sub-warp computes a partial result for an element in y. 
 * Throught shared memory some threads in a sub-warp perform parallel reduction.
 * Only the first thread in the sub_warp writes the result in the resulting matrix 
 *
 *@param M: Number of rows of the matrix A
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param X: Dense matrix
 *@param d_y: Resulting matrix
 *@param sub_warp_size: Number of threads (at most 32) that cooperates to compute an element.  
 *
 */
__global__ void ELLPACK_Sub_warp(const int M, const int K, int *nz_per_row, int *sum_nz, double *d_values, int *d_col_indices, double *d_X, double *d_y, const int sub_warp_size)
{

    __shared__ volatile double vals[MAX_BLOCK_DIM];

    const int num_elements = M * K;

    /* Thread identifier */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Global sub-warp Index */
    const int sub_warp_id = tid / sub_warp_size;

    const int lane = tid % sub_warp_size; // thread index within the sub_warp

    /* Row of the item that the warp should compute */
    const int i = sub_warp_id / K;

    /* Column of the item that the warp should compute */
    const int z = sub_warp_id % K;

    vals[threadIdx.x] = 0.0;

    if (sub_warp_id < num_elements)
    {
        const int offset = sum_nz[i];

        const int max_nz = nz_per_row[i];

        /* Partial result of matrix element Y */
        double sum = 0.0;
        for (int j = lane; j < max_nz; j += sub_warp_size)
        {
            if (d_values != NULL)
                sum += d_values[offset + j] * d_X[d_col_indices[offset + j] * K + z];
            else
                sum += 1.0 * d_X[d_col_indices[offset + j] * K + z];
        }

        vals[threadIdx.x] = sum;

        /**
         * Parallel reduction in shared memory
         */

        for (int stride = sub_warp_size >> 1; stride > 0; stride >>= 1)
            {
                if (lane < stride)
                    vals[threadIdx.x] += vals[threadIdx.x + stride];
            }

        /**
         * Only the first thread writes the result
         */

        if (lane == 0)
        {
            d_y[i * K + z] = vals[threadIdx.x];
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
 * @returns the resulting/product matrix computed by the GPU kernel
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

     FILE *f_samplings;
    /**
     * Name of the file to be created and written to
     */
    char fn[100];

    /**
     * Opening the output file
     */
    printf("Opening the output file\n");

    char *token;
    token = strtok(filename, "/");
    token = strtok(NULL, "/");

    sprintf(fn, "plots/samplings_cflush_ELLPACK_GPU_%s.csv", token);

    f_samplings = fopen(fn, "r");
    if (f_samplings == NULL){
        printf("Error opening the output file");

        f_samplings = fopen(fn, "w");
        fprintf(f_samplings, "Algorithm,K,GFLOPS\n");
    }
    else {

        fclose(f_samplings);
        f_samplings = fopen(fn, "a");
    }

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
    memory_allocation_Cuda(double, M * K, d_y);
    /* The output matrix is initialized with all zeroes */
    cudaMemset(d_y, 0, M * K * sizeof(double));
    /* Device allocation for dense matrix X */
    memory_allocation_Cuda(double, N * K, d_X);
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
    int threadsPerBlock = MAX_BLOCK_DIM;

#ifndef ELLPACK_SUB_WARP

    /* Number of blocks per grid */
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

#else 
  
    int sub_warp_size = 2;

    /**
     * Each sub warp computes an element. 
     * we need as many blocks as the number of elements to calculate divided by the number of warps per block
    */
    int warpsPerBlock = threadsPerBlock / sub_warp_size;

    /* Number of blocks per grid */
    int blocksPerGrid = (numElements + warpsPerBlock - 1) / warpsPerBlock;
#endif

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
           threadsPerBlock);

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // START TIMER
    checkCudaErrors(cudaEventRecord(start, stream));

#ifndef ELLPACK_SUB_WARP
    ELLPACK_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, K, d_nz_per_row, d_sum_nz, d_values, d_col_indices, d_X, d_y);
#else
    ELLPACK_Sub_warp<<<blocksPerGrid, threadsPerBlock>>>(M, K, d_nz_per_row, d_sum_nz, d_values, d_col_indices, d_X, d_y, sub_warp_size);
#endif

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

    double Gflops = compute_GFLOPS(K, nz, expireTimeMsec * 1e6);

    printf("GFLOPS FOR PARALLEL PRODUCT GPU: %lf\n", Gflops);

    #ifdef ELLPACK_SUB_WARP

        fprintf(f_samplings, "ellpack_sub_warp,%d,%lf\n", K, Gflops);

    #else

        fprintf(f_samplings, "ellpack,%d,%lf\n", K, Gflops);

    #endif

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

    fclose(f_samplings);

    printf("Completed parallel product ...\n");

    return h_y;
}