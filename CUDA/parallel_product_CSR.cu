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

/*------------------------------------------------------ CSR SCALAR --------------------------------------------------------------------------------------*/

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
* Each thread has the task of computing a single element of the final matrix Y.
* Item row is computed via 'thread_id / K' while column
* via 'thread_id %K'. In this version of the product the running sum is not
* optimized since we have global memory access every time we edit
* the intermediate result.
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
        if (i < M - 1 && d_irp[i] == d_irp[i + 1])
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
  * Each thread has the task of computing a single element of the final matrix Y.
  * Item row is computed via 'thread_id / K' while column
  * via 'thread_id %K'. In this version of the product the running sum is
  * optimized as we avoid continuously accessing global memory during
  * the calculation of the value of the element that the thread must compute.
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

        if (i < M - 1 && d_irp[i] == d_irp[i + 1])
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
* Each thread has the task of computing a single element of the final matrix Y.
* Item row is computed via 'thread_id / K' while column
* via 'thread_id % K'. In this version of the product we want to optimize the number of accesses to d_irp
* by storing its value in an automatic variable.
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
        int end = 0;

        if (i < M - 1)
            end = d_irp[i + 1]; // Ending index for the i-th row
        else
            end = nz;

        for (int j = start; j < end; j++)
        {
            if (d_as != NULL)
                partial_sum += d_as[j] * d_X[d_ja[j] * K + z];
            else
                partial_sum += 1.0 * d_X[d_ja[j] * K + z];
        }
        d_y[i * K + z] = partial_sum;
    }
}

/*------------------------------------------------------ CSR Vector with sub warp number of threads --------------------------------------------------------------------------------------*/

/**
 * CSR_Vector_Sub_warp - Product implementation between sparse matrix A and dense matrix X
 *
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
 *@param sub_warp_size: Number of threads (at most 32) calculating an element 
 *
 */
__global__ void CSR_Vector_Sub_warp(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, const int sub_warp_size)
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
        int start = d_irp[i];
        int end = 0;

        if (i < M - 1)
            end = d_irp[i + 1]; // Ending index for the i-th row
        else
            end = nz;

        double sum = 0.0;
        for (int j = start + lane; j < end; j += sub_warp_size)
        {

            if (d_as != NULL)
                sum += d_as[j] * d_X[d_ja[j] * K + z];
            else
                sum += 1.0 * d_X[d_ja[j] * K + z];
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

/*------------------------------------------------------ CSR Vector --------------------------------------------------------------------------------------*/

/**
 * CSR_Vector_Kernel - CSR vector implementation between sparse matrix A and dense matrix X
 *
 * Each thread in a warp computes a partial result for an element in y. 
 * Throught shared memory some threads in a warp perform parallel reduction.
 * Only the first thread in the warp writes the result in the resulting matrix 
 * 
 *@param M: Number of rows of the matrix A
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param X: Dense matrix
 *@param d_y: Resulting matrix
 *@param num_elements: Number of elements to be computed
 */
__global__ void CSR_Vector_Kernel(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y)
{
    __shared__ volatile double vals[MAX_BLOCK_DIM];

    const int num_elements = M * K;

    /* Thread identifier */
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread index

    const int lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp

    /* Global Warp Index */
    const int warp_id = tid / WARP_SIZE;

    /* Row of the item that the warp should compute */
    const int i = warp_id / K;

    /* Column of the item that the warp should compute */
    const int z = warp_id % K;

    vals[threadIdx.x] = 0.0;

    if (warp_id < num_elements)
    {
        int start = d_irp[i];
        int end = 0;

        if (i < M - 1)
            end = d_irp[i + 1];
        else
            end = nz;

        double sum = 0.0;
        for (int j = start + lane; j < end; j += WARP_SIZE)
        {

            if (d_as != NULL)
                sum += d_as[j] * d_X[d_ja[j] * K + z];
            else
                sum += 1.0 * d_X[d_ja[j] * K + z];
        }

        vals[threadIdx.x] = sum;

        // __syncthreads();
        /**
         * Parallel reduction in shared memory
         */
        if (lane < 16)
            vals[threadIdx.x] += vals[threadIdx.x + 16];

        if (lane < 8)
            vals[threadIdx.x] += vals[threadIdx.x + 8];

        if (lane < 4)
            vals[threadIdx.x] += vals[threadIdx.x + 4];

        if (lane < 2)
            vals[threadIdx.x] += vals[threadIdx.x + 2];

        if (lane < 1)
            vals[threadIdx.x] += vals[threadIdx.x + 1];

        /**
         * Only the first thread writes the result
         */

        if (lane == 0)
        {
            d_y[i * K + z] += vals[threadIdx.x];
        }
    }
}


/*------------------------------------------------------ CSR ADAPTIVE --------------------------------------------------------------------------------------*/

/**
 * CSR_Adaptive_Kernel - CSR Adaptive implementation between sparse matrix A and dense matrix X
 * 
 * CSR-Adapive is an algorithm that dynamically decides whether to use CSR-Stream or CSR-Vector
 * 
 *@param M: Number of rows of the matrix A
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param X: Dense matrix
 *@param d_y: Resulting matrix
 *@param d_rowBlocks: Array containing the starting row index per block
 *
 */
__global__ void CSR_Adaptive_Kernel(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int *d_rowBlocks)
{

    extern __shared__ volatile double LDS[];

    const int startRow = d_rowBlocks[blockIdx.x / K];

    const int nextStartRow = d_rowBlocks[(blockIdx.x / K) + 1];

    const int num_rows = nextStartRow - startRow;

    const int tid_within_block = threadIdx.x;

    const int z = blockIdx.x % K;

    LDS[tid_within_block] = 0.0;

    if (nextStartRow <= M)
    {   
        
        // If the block consists of more than one row then run CSR Stream
        if (num_rows > 1)
        {
            int nnz = 0;

            if (nextStartRow < M)
                nnz = d_irp[nextStartRow] - d_irp[startRow];
            else
                nnz = nz - d_irp[startRow];

            int first_col = d_irp[startRow];

            // Each thread writes to shared memory
            if (tid_within_block < nnz)
            {
                LDS[tid_within_block] = d_as[first_col + tid_within_block] * d_X[d_ja[first_col + tid_within_block] * K + z];
            }
            __syncthreads();

           // # numRows threads perform Scalar Reduction out of LDS to compute final output
            for (int i = startRow + tid_within_block; i < nextStartRow; i += blockDim.x)
            {

                int start = d_irp[i];
                int end = 0;

                if (i < M - 1)
                    end = d_irp[i + 1];
                else
                    end = nz;

                double temp = 0.0;
                for (int j = (start - first_col); j < (end - first_col); j++)
                {
                        temp += LDS[j];
                }

                d_y[i * K + z] = temp;
            }
        }
        // If the block consists of only one row then run CSR Vector
        else
        {
            int rowStart = d_irp[startRow];

            int rowEnd = 0;
            if (nextStartRow < M)
                rowEnd = d_irp[nextStartRow];
            else
                rowEnd = nz;

            double sum = 0.0;

            // Use all threads in a warp to accumulate multiplied elements
            for (int j = rowStart + tid_within_block; j < rowEnd; j += blockDim.x)
            {

                sum += d_as[j] * d_X[d_ja[j] * K + z];
            }

            LDS[tid_within_block] = sum;
            __syncthreads();

            // Reduce partial sums

            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
            {
                __syncthreads();
                if (tid_within_block < stride)
                    LDS[tid_within_block] += LDS[tid_within_block + stride];
            }

            // Write result
            if (tid_within_block == 0)
                d_y[startRow * K + z] = LDS[tid_within_block];
        }
    }
}


/**
 * csr_adaptive_rowblocks: This CPU code calculates the number of rows of a CSR matrix that can fit in the shared memory
 *
 *
 * @param M: Number of rows
 * @param nz: Number of non-zeroes
 * @param irp: Vector containing the column index of the first nonzero of rows
 * @param rowBlocks: Array containing the starting row index per block
 * @param threadsPerBlock: pointer to an integer representing the computed threads per block
 *
 * Returns the number of blocks computed
 *
 * */

int csr_adaptive_rowblocks(int M, int nz, int *irp, int **rowBlocks, int *threadsPerBlock)
{

    all_zeroes_memory_allocation(int, M, *rowBlocks);

    (*rowBlocks)[0] = 0;
    int sum_nz = 0, last_i = 0, ctr = 1;

    // int sh_memory_per_block = 49152;                     // Total amount shared memory per block in bytes
    // int max_size = sh_memory_per_block / sizeof(double); // Total amount of double in the shared memory

    // int local_size = max_size;

    // if (local_size % WARP_SIZE != 0)
    //     local_size += WARP_SIZE - (local_size % WARP_SIZE);

    // if (local_size > MAX_BLOCK_DIM)
    //     local_size = MAX_BLOCK_DIM;

    // Computing the average number of non-zeroes per row 
    int local_size = pow(2,floor(log2((nz + M - 1)/ M)));
    if (local_size < WARP_SIZE) local_size = WARP_SIZE;
    if (local_size > MAX_BLOCK_DIM) local_size = MAX_BLOCK_DIM;

    for (int i = 1; i <= M; i++)
    {
        if (i == M)
            sum_nz += nz - irp[i - 1];
        else
            sum_nz += irp[i] - irp[i - 1]; // Count non-zeroes in this row

        if (sum_nz == local_size)
        { // The row fills up to LOCAL SIZE
            last_i = i;
            (*rowBlocks)[ctr++] = i;
            sum_nz = 0;
        }
        else if (sum_nz > local_size)
        {

            if (i - last_i > 1)
            {
                (*rowBlocks)[ctr++] = i - 1; // this extra row will not fit
                --i;
            }
            else if (i - last_i == 1)
            {
                // This row is too large. It has too much zeroes
                (*rowBlocks)[ctr++] = i;
            }

            last_i = i;
            sum_nz = 0;
        }
    }

    (*rowBlocks)[ctr++] = M;
    *threadsPerBlock = local_size;

    return ctr;
}

/*------------------------------------------------------ KERNEL SETUP --------------------------------------------------------------------------------------*/

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

double *CSR_GPU(int M, int N, int K, int nz, double *h_as, int *h_ja, int *h_irp, double **X)
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

#ifdef CSR_ADAPTIVE
    int *rowBlocks = NULL;
    int *d_rowBlocks = NULL;
#endif

    /* 2D to 1D dense matrix X conversion*/
    h_X = convert_2D_to_1D(N, K, X);

    /* Y array host memory allocation */
    memory_allocation(double, M *K, h_y);

    printf("Allocating device variables for CPU CSR product ...\n");

    /* Y array host memory allocation */
    memory_allocation_Cuda(double, M *K, d_y);
    /* Device allocation for dense matrix X */
    memory_allocation_Cuda(double, N *K, d_X);
    if (h_as != NULL)
        /* Device allocation for the as vector containing non-zero elements */
        memory_allocation_Cuda(double, nz, d_as);

    /* Device allocation for the as vector containing non-zero elements */
    memory_allocation_Cuda(int, nz, d_ja);
    /* Device allocation for the irp vector containing the pointer to the vector entry ja */
    memory_allocation_Cuda(int, M, d_irp);

    printf("Copy input data from the host memory to the CUDA device\n");

    if (h_as != NULL)
        /* Copy of the contents of the vector as from the Host to the Device */
        memcpy_to_dev(h_as, d_as, double, nz);

    /* Copy of the contents of the vector ja from the Host to the Device */
    memcpy_to_dev(h_ja, d_ja, int, nz);
    /* Copy of the contents of the vector irp from the Host to the Devicee */
    memcpy_to_dev(h_irp, d_irp, int, M);
    /* Copy of the dense vector X from the Host to the Device*/
    memcpy_to_dev(h_X, d_X, double, N *K);

    /* Number of threads per block */
    int threadsPerBlock = MAX_BLOCK_DIM;

#ifdef CSR_ADAPTIVE

    int number_of_blocks = csr_adaptive_rowblocks(M, nz, h_irp, &rowBlocks, &threadsPerBlock);

    // /* Device allocation for d_rowBlocks */
    memory_allocation_Cuda(int, number_of_blocks, d_rowBlocks);

    // /* Copy rowBlocks from the Host to the Device*/
    memcpy_to_dev(rowBlocks, d_rowBlocks, int, number_of_blocks);

    /* Number of blocks per grid */
    int blocksPerGrid = (number_of_blocks - 1) * K;

#elif CSR_VECTOR

    /* Number of elements of the product matrix Y */
    int numElements = M * K;

    int warpsPerBlock = threadsPerBlock / WARP_SIZE; //<-- Per original CSR_vector

    // int sub_warp_size = pow(2,floor(log2((nz + M - 1)/ M)));
    // if (sub_warp_size > WARP_SIZE) sub_warp_size = WARP_SIZE;

    // int warpsPerBlock = threadsPerBlock / sub_warp_size; //<-- Per CSR_vector_sub_warp

    /* Number of blocks per grid */
    int blocksPerGrid = (numElements + warpsPerBlock - 1) / warpsPerBlock;
    
#else

    /* Number of elements of the product matrix Y */
    int numElements = M * K;

    /* Number of blocks per grid */
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

#endif

    printf("CUDA kernel for K = %d launch with %d blocks of %d threads\n", K, blocksPerGrid,
           threadsPerBlock);

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // START TIMER
    checkCudaErrors(cudaEventRecord(start, stream));

#ifdef CSR_ADAPTIVE

    /* CSR Adaptive */
    CSR_Adaptive_Kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, d_rowBlocks);

    //CSR_Adaptive_Kernel_v2<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, d_rowBlocks);

#elif CSR_VECTOR
    //  /* CSR Vector */
    CSR_Vector_Kernel<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y);

    //CSR_Vector_Sub_warp<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, sub_warp_size);

#else

    /* Versione accesso alla memoria globale non ottimizzato */
    // CSR_kernel_v1<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, numElements);

    // CSR_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, numElements);

    /* Versione accesso alla memoria globale ottimizzato */
    CSR_kernel_v3<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, numElements);

#endif // CSR_ADAPTIVE

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

    printf("GFLOPS FOR PARALLEL PRODUCT GPU: %lf\n", compute_GFLOPS(K, nz, expireTimeMsec * 1e6));

    printf("Copy output data from the CUDA device to the host memory\n");

    /* Copy of the product matrix Y from the Device to the Host */
    memcpy_to_host(d_y, h_y, double, M *K);

    /* Start the memory cleaning process on Device */
    printf("Freeing Device memory ...\n");
    if (h_as != NULL)
        free_memory_Cuda(d_as);
    free_memory_Cuda(d_ja);
    free_memory_Cuda(d_irp);
    free_memory_Cuda(d_X);
    free_memory_Cuda(d_y);

#ifdef CSR_ADAPTIVE
    free_memory_Cuda(d_rowBlocks);
#endif

    /* Start the memory cleaning process on Host */

    printf("Freeing host memory ...\n");

    //print_y_GPU(M, K, h_y);

    free(h_X);

#ifdef CSR_ADAPTIVE
    free(rowBlocks);
#endif

    printf("Completed parallel product CSR ...\n");

    return h_y;
}