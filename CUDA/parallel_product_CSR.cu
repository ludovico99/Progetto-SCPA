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
 * CSR_Scalar_v1 -  Product implementation between sparse matrix A and dense matrix X
 *
 *@param M: Number of rows of the matrix A
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param X: Dense matrix
 *@param d_y: Resulting matrix
 *
 * Each thread has the task of computing a single element of the final matrix Y.
 * Item row is computed via 'thread_id / K' while column
 * via 'thread_id %K'. In this version of the product the running sum is not
 * optimized since we have global memory access every time we edit
 * the intermediate result.
 */
__global__ void CSR_Scalar_v1(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y)
{
    /* Thread identifier */
    const long tid = blockDim.x * blockIdx.x + threadIdx.x;

    const long num_elements = M * K;

    /* Row of the item that the thread should compute */
    const int i = tid / K;

    /* Item column that the thread should compute */
    const int z = tid % K;

    if (tid < num_elements)
    {
        if (i < M - 1 && d_irp[i] == d_irp[i + 1])
        {
            d_y[i * K + z] = 0.0;
        }
        else
        {
            for (int j = d_irp[i]; j < d_irp[i + 1]; j++)
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
 * CSR_Scalar_v2 - Product implementation between sparse matrix A and dense matrix X
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
__global__ void CSR_Scalar_v2(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y)
{
    /* Thread identifier */
    const long tid = blockDim.x * blockIdx.x + threadIdx.x;

    const long num_elements = M * K;

    /* Row of the item that the thread should compute */
    const int i = tid / K;

    /* Item column that the thread should compute */
    const int z = tid % K;

    /* Partial result of matrix element Y */
    double partial_sum = 0.0;

    if (tid < num_elements)
    {

        if (i < M - 1 && d_irp[i] == d_irp[i + 1])
        {
            d_y[i * K + z] = 0.0;
        }
        else
        {

            for (int j = d_irp[i]; j < d_irp[i + 1]; j++)
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
 * CSR_Scalar_v3 - Product implementation between sparse matrix A and dense matrix X
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
__global__ void CSR_Scalar_v3(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y)
{
    /* Thread identifier */
    const long tid = blockDim.x * blockIdx.x + threadIdx.x;

    const long num_elements = M * K;

    /* Row of the item that the thread should compute */
    int i = tid / K;

    /* Item column that the thread should compute */
    int z = tid % K;

    /* Partial result of matrix element Y */
    double partial_sum = 0.0;

    if (tid < num_elements)
    {
        int start = d_irp[i];   // Starting index for the i-th row
        int end = d_irp[i + 1]; // Ending index for the i-th row

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
    const long tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Global sub-warp Index */
    const long sub_warp_id = tid / sub_warp_size;

    /* Thread index within the sub_warp */
    const int lane = tid % sub_warp_size;

    /* Row of the item that the warp should compute */
    const int i = sub_warp_id / K;

    /* Column of the item that the warp should compute */
    const int z = sub_warp_id % K;

    vals[threadIdx.x] = 0.0;

    if (sub_warp_id < num_elements)
    {
        int start = d_irp[i];   // Starting index for the i-th row
        int end = d_irp[i + 1]; // Ending index for the i-th row

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

/*------------------------------------------------------ CSR Vector by row --------------------------------------------------------------------------------------*/

/**
 * CSR_Vector_by_row - CSR vector implementation between sparse matrix A and dense matrix X
 * Each warp is responsible for calculating all components of a row
 * Each sub-warp calculates an element in y for the row assigned to the warp of which the sub-warp is a part
 * Throught shared memory some threads in a sub-warp perform parallel reduction.
 * Only the first thread in the sub-warp writes the result in the resulting matrix
 *
 *@param M: Number of rows of the matrix A
 *@param N: Number of columns of the matrix A, Number of rows of the matrix X
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param d_X: Dense matrix
 *@param d_y: Resulting matrix
 */

__global__ void CSR_Vector_by_row(const int M, const int N, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y)
{
    __shared__ volatile double vals[MAX_BLOCK_DIM];

    /* Thread identifier */
    const long tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Thread index within the warp*/
    const int tid_within_warp = threadIdx.x & (WARP_SIZE - 1); //It's a value between 0 to 31

    /* Global Warp Index */
    const long warp_id = tid / WARP_SIZE;

    /* Row of the item that the warp should compute */
    const int i = warp_id;

    /* Number of threads in a sub_warp*/
    const int sub_warp_size = 4;

    /* Sub-warp index in the warp*/
    const int sub_warp_id = tid_within_warp / sub_warp_size;

    /* Thread index within a sub-warp*/
    const int tid_within_sub_warp = tid % sub_warp_size;

    /*Number of sub-warp in a warp*/
    const int num_sub_warps = WARP_SIZE / sub_warp_size; // It's 8

    if (i < M)
    {

        int start = d_irp[i];
        int end = d_irp[i + 1];

        for (int z = sub_warp_id; z < K; z += num_sub_warps)
        {

            /*Starting column index in the 1D vector d_X*/
            const long col = z * N;

            double sum = 0.0;
            for (int j = start + tid_within_sub_warp; j < end; j += sub_warp_size)
            {

                if (d_as != NULL)
                    sum += d_as[j] * d_X[col + d_ja[j]];
                else
                    sum += 1.0 * d_X[col + d_ja[j]];
            }
            vals[threadIdx.x] = sum;

            /**
             * Parallel reduction in shared memory
             */

            for (int stride = sub_warp_size >> 1; stride > 0; stride >>= 1)
            {
                if (tid_within_sub_warp < stride)
                    vals[threadIdx.x] += vals[threadIdx.x + stride];
            }

            /**
             * Only the first thread writes the result
             */

            if (tid_within_sub_warp == 0)
            {
                d_y[i * K + z] = vals[threadIdx.x];
            }
        }
    }
}

/*------------------------------------------------------ CSR Vector --------------------------------------------------------------------------------------*/

/**
 * CSR_Vector - CSR vector implementation between sparse matrix A and dense matrix X
 *
 * Each thread in a warp computes a partial result for an element in y.
 * Throught shared memory some threads in a warp perform parallel reduction.
 * Only the first thread in the warp writes the result in the resulting matrix
 *
 *@param M: Number of rows of the matrix A
 *@param N: Number of columns of the matrix A, Number of rows of the matrix X
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param d_X: Dense matrix
 *@param d_y: Resulting matrix
 */

__global__ void CSR_Vector(const int M, const int N, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y)
{
    __shared__ volatile double vals[MAX_BLOCK_DIM];

    const int num_elements = M * K;

    /* Thread identifier */
    const long tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread index

    const int lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp

    /* Global Warp Index */
    const long warp_id = tid / WARP_SIZE;

    /* Row of the item that the warp should compute */
    const int i = warp_id / K;

    /* Column of the item that the warp should compute */
    const int z = warp_id % K;

    /*Starting */
    const long col = z * N;

    if (warp_id < num_elements)
    {

        int start = d_irp[i];
        int end = d_irp[i + 1];

        double sum = 0.0;
        for (int j = start + lane; j < end; j += WARP_SIZE)
        {

            if (d_as != NULL)
                sum += d_as[j] * d_X[col + d_ja[j]];
            else
                sum += 1.0 * d_X[col + d_ja[j]];
        }

        vals[threadIdx.x] = sum;

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
            d_y[i * K + z] = vals[threadIdx.x];
        }
    }
}



/*------------------------------------------------------ CSR Vector RIDUZIONE OTTIMIZZATA --------------------------------------------------------------------------------------*/

/**
 * CSR_Vector - CSR vector implementation between sparse matrix A and dense matrix X
 *
 * Each thread in a warp computes a partial result for an element in y.
 * Throught shared memory some threads in a warp perform parallel reduction.
 * Only the first thread in the warp writes the result in the resulting matrix
 *
 *@param M: Number of rows of the matrix A
 *@param N: Number of columns of the matrix A, Number of rows of the matrix X
 *@param K:  Number of columns of the matrix X
 *@param nz: Number of nz
 *@param d_as: Vector containing the non-zero elements of the sparse array
 *@param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 *@param d_irp: Vector containing the column index of the first nonzero of rows
 *@param d_X: Dense matrix
 *@param d_y: Resulting matrix
 */

__global__ void CSR_Vector_optimazed_reduction(const int M, const int N, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y)
{
    const int num_elements = M * K;

    /* Thread identifier */
    const long tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread index

    const int lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp

    /* Global Warp Index */
    const long warp_id = tid / WARP_SIZE;

    /* Row of the item that the warp should compute */
    const int i = warp_id / K;

    /* Column of the item that the warp should compute */
    const int z = warp_id % K;

    /*Starting */
    const long col = z * N;

    if (warp_id < num_elements)
    {

        int start = d_irp[i];
        int end = d_irp[i + 1];

        double sum = 0.0;
        for (int j = start + lane; j < end; j += WARP_SIZE)
        {

            if (d_as != NULL)
                sum += d_as[j] * d_X[col + d_ja[j]];
            else
                sum += 1.0 * d_X[col + d_ja[j]];
        }

        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset, 32);        

        /**
         * Only the first thread writes the result
         */

        if (lane == 0)
        {
            d_y[i * K + z] = sum;
        }
    }
}


/*------------------------------------------------------ CSR ADAPTIVE --------------------------------------------------------------------------------------*/

/**
 * CSR_Adaptive - CSR Adaptive implementation between sparse matrix A and dense matrix X
 *
 * CSR-Adapive is an algorithm that dynamically decides whether to use CSR-Stream or CSR-Vector
 *
 *@param M: Number of rows of the matrix A
 *@param N: Number of columns of the matrix A, Number of rows of the matrix X
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
__global__ void CSR_Adaptive(const int M, const int N, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int *d_rowBlocks)
{

    extern __shared__ volatile double LDS[];

    /*my_block identifies in d_rowBlocks the rows of its competence*/
    const int my_block = blockIdx.x / K;

    /*Starting row index for the current block*/
    const int startRow = d_rowBlocks[my_block];

    /*Index of the starting row of the next row for the current block*/
    const int nextStartRow = d_rowBlocks[my_block + 1];
    
    /* Number of rows in the current block*/
    const int num_rows = nextStartRow - startRow;

    const int tid_within_block = threadIdx.x;

    /*Columns assigned to the current block*/
    const int z = blockIdx.x % K;

    if (nextStartRow <= M)
    {

        // If the block consists of more than one row then run CSR Stream
        if (num_rows > 1)
        {
            int nnz = 0;

            nnz = d_irp[nextStartRow] - d_irp[startRow];

            int first_col = d_irp[startRow];

            // Each thread writes to shared memory
            if (tid_within_block < nnz)
            {
                if (d_as != NULL)
                    LDS[tid_within_block] = d_as[first_col + tid_within_block] * d_X[z * N + d_ja[first_col + tid_within_block]];
                else
                    LDS[tid_within_block] = 1.0 * d_X[z * N + d_ja[first_col + tid_within_block]];
            }
            __syncthreads();

            // # numRows threads perform Scalar Reduction out of LDS to compute final output
            for (int i = startRow + tid_within_block; i < nextStartRow; i += blockDim.x)
            {

                int start = d_irp[i];
                int end = d_irp[i + 1];

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

            int rowEnd = d_irp[nextStartRow];

            double sum = 0.0;

            // Use all threads in a warp to accumulate multiplied elements
            for (int j = rowStart + tid_within_block; j < rowEnd; j += blockDim.x)
            {
                if (d_as != NULL)
                    sum += d_as[j] * d_X[z * N + d_ja[j]];
                else
                    sum += 1.0 * d_X[z * N + d_ja[j]];
            }

            LDS[tid_within_block] = sum;
            __syncthreads();

            // Reduce partial sums

            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
            {
                if (tid_within_block < stride)
                    LDS[tid_within_block] += LDS[tid_within_block + stride];

                __syncthreads();
            }

            // Write result
            if (tid_within_block == 0)
                d_y[startRow * K + z] = LDS[tid_within_block];
        }
    }
}

/*------------------------------------------------------ CSR ADAPTIVE SUB_BLOCKS --------------------------------------------------------------------------------------*/

/**
 * CSR_Adaptive_sub_blocks - CSR Adaptive (variant) implementation between sparse matrix A and dense matrix X
 *
 * CSR-Adapive is an algorithm that dynamically decides whether to use CSR-Stream or CSR-Vector.
 * Each kernel computes the i-th row in the resulting matrix.
 * The shared memory has size MAX_BLOCK_DIM = 512.
 * For K = 64 each block has at most 512 / 64 = 8 non-zeroes if CSR stream should be run.
 * There are 8 threads calculating the k-th element of the i-th row for every k from 0 to 63
 *
 *@param M: Number of rows of the matrix A
 *@param N: Number of columns of the matrix A, Number of rows of the matrix X
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
__global__ void CSR_Adaptive_sub_blocks(const int M, const int N, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int *d_rowBlocks)
{

    extern __shared__ volatile double LDS[];

    const int my_block = blockIdx.x;

    const int startRow = d_rowBlocks[my_block];

    const int nextStartRow = d_rowBlocks[my_block + 1];

    const int num_rows = nextStartRow - startRow;

    /* Number of threads in a sub-block*/
    const int sub_block_size = blockDim.x / K; // For K = 64, sub_block_size = 8
    /*Local sub-block Index*/
    const int sub_block_id = threadIdx.x / sub_block_size; // For K = 64, sub_block_id = [0 - 63]
    /* Thread index within a sub-block*/
    const int tid_within_sub_block = threadIdx.x % sub_block_size; // For K = 64, tid_within_sub_block = [0 - 7]
    /*Number of sub-block in a block*/
    const int num_sub_blocks = blockDim.x / sub_block_size; // For K = 64, num_sub_blocks = 64

    if (nextStartRow <= M)
    {

        // If the block consists of more than one row then run CSR Stream
        if (num_rows > 1)
        {
            int nnz = 0;

            nnz = d_irp[nextStartRow] - d_irp[startRow];

            int first_col = d_irp[startRow];

            for (int z = sub_block_id; z < K; z += num_sub_blocks)
            {
                // Each thread writes to shared memory
                if (tid_within_sub_block < nnz)
                {
                    if (d_as != NULL)
                        LDS[threadIdx.x] = d_as[first_col + tid_within_sub_block] * d_X[z * N + d_ja[first_col + tid_within_sub_block]];
                    else
                        LDS[threadIdx.x] = 1.0 * d_X[z * N + d_ja[first_col + tid_within_sub_block]];
                }
                __syncthreads();

                // sub_block_size threads perform Scalar Reduction out of LDS to compute final output
                for (int i = startRow + tid_within_sub_block; i < nextStartRow; i += sub_block_size)
                {

                    int start = d_irp[i];
                    int end = d_irp[i + 1];

                    double temp = 0.0;
                    for (int j = (start - first_col); j < (end - first_col); j++)
                    {
                        temp += LDS[j + z * sub_block_size];
                    }

                    d_y[i * K + z] = temp;
                }
            }
        }
        // If the block consists of only one row then run CSR Vector
        else
        {
            int rowStart = d_irp[startRow];

            int rowEnd = d_irp[nextStartRow];

            for (int z = sub_block_id; z < K; z += num_sub_blocks)
            {
                double sum = 0.0;

                // Use all threads in a warp to accumulate multiplied elements
                for (int j = rowStart + tid_within_sub_block; j < rowEnd; j += sub_block_size)
                {
                    if (d_as != NULL)
                        sum += d_as[j] * d_X[z * N + d_ja[j]];
                    else
                        sum += 1.0 * d_X[z * N + d_ja[j]];
                }

                LDS[threadIdx.x] = sum;
                __syncthreads();

                // Reduce partial sums

                for (int stride = sub_block_size >> 1; stride > 0; stride >>= 1)
                {
                    __syncthreads();
                    if (tid_within_sub_block < stride)
                        LDS[tid_within_sub_block + z * sub_block_size] += LDS[tid_within_sub_block + z * sub_block_size + stride];
                }

                // Write result
                if (tid_within_sub_block == 0)
                    d_y[startRow * K + z] = LDS[tid_within_sub_block + z * sub_block_size];
            }
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
 * @returns the number of blocks computed
 *
 * */

int csr_adaptive_rowblocks(int M, int K, int nz, int *irp, int **rowBlocks, int *threadsPerBlock, int mode)
{

    int sum_nz = 0, last_i = 0, ctr = 1, local_size;
    all_zeroes_memory_allocation(int, M + 1, *rowBlocks);

    (*rowBlocks)[0] = 0;

    if (mode == adaptive_sub_blocks) 
        local_size = MAX_BLOCK_DIM / K; 
    else
    {

        // int sh_memory_per_block = 49152;                     // Total amount shared memory per block in bytes
        // int max_size = sh_memory_per_block / sizeof(double); // Total amount of double in the shared memory

        /*Computing the average number of non-zeroes per row*/
        local_size = pow(2, floor(log2((nz + M - 1) / M)));

        if (local_size < WARP_SIZE)
            local_size = WARP_SIZE;
        if (local_size > MAX_BLOCK_DIM)
            local_size = MAX_BLOCK_DIM; // Surely MAX_BLOCK_DIM * 8 bytes is less than sh_memory_per_block bytes
    }

    for (int i = 1; i <= M; i++)
    {
        sum_nz += irp[i] - irp[i - 1]; // Count non-zeroes in the i-th row

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
        else if (i == M)
            (*rowBlocks)[ctr++] = M; // Inserting last start row if it has not been previuosly inserted
    }
    
    if (mode == adaptive_sub_blocks)
        *threadsPerBlock = MAX_BLOCK_DIM;
    else
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
 * @returns the resulting/product matrix computed by the GPU kernel
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

#if defined(CSR_ADAPTIVE) || defined(CSR_ADAPTIVE_SUB_BLOCK)
    int *rowBlocks = NULL;
    int *d_rowBlocks = NULL;
#endif

#if defined(CSR_ADAPTIVE) || defined(CSR_ADAPTIVE_SUB_BLOCK) || defined(CSR_VECTOR) || defined(CSR_VECTOR_BY_ROW)
    h_X = transpose_from_2D(N, K, X);
#else
    /* 2D to 1D dense matrix X conversion*/
    h_X = convert_2D_to_1D(N, K, X);
#endif

    /* Y array host memory allocation */
    memory_allocation(double, M *K, h_y);

    printf("Allocating device variables for CPU CSR product ...\n");

    /* Y array host memory allocation */
    memory_allocation_Cuda(double, M *K, d_y);

    /* The output matrix is initialized with all zeroes */
    cudaMemset(d_y, 0, M * K * sizeof(double));

    /* Device allocation for dense matrix X */
    memory_allocation_Cuda(double, N *K, d_X);

    if (h_as != NULL)
        /* Device allocation for the as vector containing non-zero elements */
        memory_allocation_Cuda(double, nz, d_as);

    /* Device allocation for the as vector containing non-zero elements */
    memory_allocation_Cuda(int, nz, d_ja);
    /* Device allocation for the irp vector containing the pointer to the vector entry ja */
    memory_allocation_Cuda(int, M + 1, d_irp);

    printf("Copy input data from the host memory to the CUDA device\n");

    if (h_as != NULL)
        /* Copy of the contents of the vector as from the Host to the Device */
        memcpy_to_dev(h_as, d_as, double, nz);

    /* Copy of the contents of the vector ja from the Host to the Device */
    memcpy_to_dev(h_ja, d_ja, int, nz);
    /* Copy of the contents of the vector irp from the Host to the Device */
    memcpy_to_dev(h_irp, d_irp, int, M + 1);
    /* Copy of the dense vector X from the Host to the Device*/
    memcpy_to_dev(h_X, d_X, double, N *K);

    /* Number of threads per block */
    int threadsPerBlock = MAX_BLOCK_DIM;

#ifdef CSR_ADAPTIVE

    int number_of_blocks = csr_adaptive_rowblocks(M, K, nz, h_irp, &rowBlocks, &threadsPerBlock, adaptive);

    /* Device allocation for d_rowBlocks */
    memory_allocation_Cuda(int, number_of_blocks, d_rowBlocks);

    /* Copy rowBlocks from the Host to the Device*/
    memcpy_to_dev(rowBlocks, d_rowBlocks, int, number_of_blocks);

    /* Number of blocks per grid */
    int blocksPerGrid = (number_of_blocks - 1) * K;

#elif CSR_ADAPTIVE_SUB_BLOCK

    int number_of_blocks = csr_adaptive_rowblocks(M, K, nz, h_irp, &rowBlocks, &threadsPerBlock, adaptive_sub_blocks);

    // /* Device allocation for d_rowBlocks */
    memory_allocation_Cuda(int, number_of_blocks, d_rowBlocks);

    // /* Copy rowBlocks from the Host to the Device*/
    memcpy_to_dev(rowBlocks, d_rowBlocks, int, number_of_blocks);

    /* Number of blocks per grid */
    int blocksPerGrid = number_of_blocks - 1;

#elif CSR_VECTOR
    /* Number of elements of the product matrix Y */
    int numElements = M * K;

    int warpsPerBlock = threadsPerBlock / WARP_SIZE; //<-- Per original CSR_vector

    /* Number of blocks per grid */
    int blocksPerGrid = (numElements + warpsPerBlock - 1) / warpsPerBlock;

#elif CSR_VECTOR_BY_ROW

    int warpsPerBlock = threadsPerBlock / WARP_SIZE;

    /* Number of blocks per grid */
    int blocksPerGrid = (M + warpsPerBlock - 1) / warpsPerBlock;

#elif CSR_VECTOR_SUB_WARP

    /* Number of elements of the product matrix Y */
    int numElements = M * K;

    /**
    * sub_warp_size is the power of 2 closest to the mean (rounded down) of non-zeros per row
    */

    // int sub_warp_size = pow(2,floor(log2((nz + M - 1)/ M)));
    // if (sub_warp_size > WARP_SIZE) sub_warp_size = WARP_SIZE;

    int sub_warp_size = 2;
    int warpsPerBlock = threadsPerBlock / sub_warp_size;

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
    CSR_Adaptive<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(M, N, K, nz, d_as, d_ja, d_irp, d_X, d_y, d_rowBlocks);

#elif CSR_ADAPTIVE_SUB_BLOCK

    CSR_Adaptive_sub_blocks<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(M, N, K, nz, d_as, d_ja, d_irp, d_X, d_y, d_rowBlocks);

#elif CSR_VECTOR

    /* CSR Vector */
    //CSR_Vector<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, nz, d_as, d_ja, d_irp, d_X, d_y);
    CSR_Vector_optimazed_reduction<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, nz, d_as, d_ja, d_irp, d_X, d_y);

#elif CSR_VECTOR_BY_ROW

    CSR_Vector_by_row<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, nz, d_as, d_ja, d_irp, d_X, d_y);

#elif CSR_VECTOR_SUB_WARP

    CSR_Vector_Sub_warp<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, sub_warp_size);
#else

    /* Versione accesso alla memoria globale non ottimizzato */
    // CSR_Scalar_v1<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y);

    // CSR_Scalar_v2<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y);

    /* Versione accesso alla memoria globale ottimizzato */
    CSR_Scalar_v3<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y);

#endif

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

#if defined(CSR_ADAPTIVE) || defined(CSR_ADAPTIVE_SUB_BLOCK)
    free_memory_Cuda(d_rowBlocks);
#endif

    /* Start the memory cleaning process on Host */

    printf("Freeing host memory ...\n");

    // print_y_GPU(M, K, h_y);

    free(h_X);

#if defined(CSR_ADAPTIVE) || defined(CSR_ADAPTIVE_SUB_BLOCK)
    free(rowBlocks);
#endif

    printf("Completed parallel product CSR ...\n");

    return h_y;
}