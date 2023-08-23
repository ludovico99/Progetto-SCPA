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

int THR = 53;

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

        /* Only the first thread writes the result */

        if (lane == 0)
        {
            d_y[i * K + z] = vals[threadIdx.x];
        }
    }
}

/**
 * CSR_Adaptive_personalizzato - CSR Adaptive (variant) implementation between sparse matrix A and dense matrix X
 *
 * @param M: Number of rows of the matrix A
 * @param N: Number of columns of the matrix A, Number of rows of the matrix X
 * @param K:  Number of columns of the matrix X
 * @param nz: Number of nz
 * @param d_as: Vector containing the non-zero elements of the sparse array
 * @param d_ja: Vector containing the column indexes of the nonzero elements of the sparse array
 * @param d_irp: Vector containing the column index of the first nonzero of rows
 * @param X: Dense matrix
 * @param d_y: Resulting matrix
 * @param d_metadata:
 * @param d_items_scalar:
 * @param d_items_vector:
 *
 */
__global__ void CSR_Adaptive_personalizzato(const int M, const int N, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, long *d_metadata, struct item* d_items_scalar, struct item* d_items_vector)
{
    int start;
    int end;
    long tid;
    long block_id;
    double partial_sum;
    struct item elem;

    /* Identificativo globale del thread */
    tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Identificativo del blocco */
    block_id = blockIdx.x;

    /* Risultato parziale per l'elemento della matrice Y */
    partial_sum=0;

    if(block_id < d_metadata[3])
    {
        //CSR SCALAR

        /* Verifico se questo thread deve effettivamente computare un elemento con CSR SCALAR */
        if(tid < d_metadata[1])
        {
            /* Struttura dati che contiene la riga e la colonna dell'elemento Yij che deve essere computato */
            elem=d_items_scalar[tid];

            start=d_irp[elem.row];
            end=d_irp[elem.row+1];

            for (int j = start; j < end; j++)
            {
                if (d_as != NULL)
                    partial_sum += d_as[j] * d_X[d_ja[j] * K + elem.col];
                else
                    partial_sum += 1.0 * d_X[d_ja[j] * K + elem.col];
            }   

            d_y[elem.row * K + elem.col] = partial_sum;
        }
    }
    else
    {
        //VECTOR SUB WARP

        /* Verifico se questo thread deve effettivamente computare un elemento con VECTOR SUB WARP */
        if((tid - MAX_BLOCK_DIM * d_metadata[3])< d_metadata[2])
        {
            __shared__ volatile double vals[MAX_BLOCK_DIM];

            /* Indice del thread all'interno del sub warp */
            const int lane= tid % SUB_WARP_SIZE;

            /* Struttura dati che contiene la riga e la colonna dell'elemento Yij che deve essere computato */
            elem=d_items_vector[tid - MAX_BLOCK_DIM * d_metadata[3]];

            vals[threadIdx.x]=0.0;

            start=d_irp[elem.row];
            end=d_irp[elem.row+1];  

            double sum=0.0;   

            for (int j = start + lane; j < end; j+=SUB_WARP_SIZE)
            {
                if (d_as != NULL)
                    sum += d_as[j] * d_X[d_ja[j] * K + elem.col];
                else
                    sum += 1.0 * d_X[d_ja[j] * K + elem.col];
            }         

            vals[threadIdx.x]=sum;

            /**
            * Parallel reduction in shared memory
            */

            for (int stride = SUB_WARP_SIZE >> 1; stride > 0; stride >>= 1)
            {
                if (lane < stride)
                    vals[threadIdx.x] += vals[threadIdx.x + stride];
            }

            /**
            * Only the first thread writes the result
            */

            if (lane == 0)
            {
                d_y[elem.row * K + elem.col] = vals[threadIdx.x];
            }

        }
    }
}


/**
 * csr_adaptive_personalizzato_number_of_blocks: This CPU code calculates the number of blocks per Grid
 * 
 * @param nz_per_row: Number of NZ per row
 * @param threadsPerBlock: pointer to an integer representing the computed threads per block
 * 
 * @returns The number of blocks computed
*/
struct core_adaptive_personalizzato *csr_adaptive_personalizzato_number_of_blocks(int M, int *nz_per_row, int threadsPerBlock, int K)
{
    int i;
    long *metadata;
    long total_row_scalar;
    long total_row_vector;
    long total_threads;
    long number_of_blocks_per_Grid;
    long total_block_scalar;
    long total_block_vector;
    struct item *items_scalar;
    struct item *items_vector;
    struct core_adaptive_personalizzato *ret;


    /* Numero totale di threads, sia quelli che utilizzano CSR SCALAR che quelli che utilizzano VECTOR SUB WARP */
    total_threads=0;

    /* Numero totale di righe che hanno un numero di non zeri inferiore della threshold */
    total_row_scalar=0;

    /* Numero totale di righe che hanno un numero di non zeri superiore della threshold */
    total_row_vector=0;

    /* Numero totale di blocchi nella griglia considerando sia quelli relativi a CSR SCALAR che quelli relativi a VECTOR SUB WARP */
    number_of_blocks_per_Grid=0;

    /*
     * Computo il numero di righe che hanno un numero di NZ maggiore e minore della threshold.
     * Se il numero di NZ è minore della threshold allora viene eseguito il CSR Scalar;
     * altrimenti, viene eseguito il CSR Sub-Vector.
     * Computo il numero totale di threads considerando che il CSR SCALAR necessita di
     * un solo thread per calcolare un elemento della matrice Y mentre il VECTOR SUB WARP
     * necessita di SUB_WARP_SIZE threads per calcolare un elemento della matrice Y.
     */

    for(i=0; i<M; i++)
    {
        if(nz_per_row[i] < THR)
        {
            //CSR-Scalar
            total_threads+=1*K;
            total_row_scalar++;
        }            
        else
        {
            //CSR-Sub-Vector
            total_threads+=SUB_WARP_SIZE*K;
            total_row_vector++;
        }
    }

    items_scalar=(struct item *)malloc(sizeof(struct item)*total_row_scalar*1*K);
    if(items_scalar==NULL) return NULL;

    items_vector=(struct item *)malloc(sizeof(struct item)*total_row_vector*SUB_WARP_SIZE*K);
    if(items_vector==NULL) return NULL;

    /* Contatore per spiazzarmi al'interno dell'array items_scalar */
    int counter_scalar=0;

    /* Contatore per spiazzarmi al'interno dell'array items_vector */
    int counter_vector=0;

    /*
     * Determino quali sono gli elementi che ogni thread deve computare. Tutto ciò viene
     * fatto prima di lanciare i kernels poiché la computazione degli elementi della 
     * matrice Y è stata distribuita tra i due algoritmi CSR SCALAR e VECTOR SUB WARP.
     */
    for(i=0; i<M; i++)
    {
        if(nz_per_row[i] < THR)
        {
            //CSR-Scalar
            for(int j=0; j<K; j++)
            {
                items_scalar[counter_scalar].row = i;
                items_scalar[counter_scalar].col = j;
                counter_scalar++;
            }
        }            
        else
        {
            //CSR-Sub-Vector
            for(int j=0; j<K; j++)
            {
                for(int sub=0; sub<SUB_WARP_SIZE;sub++)
                {
                    items_vector[counter_vector].row = i;
                    items_vector[counter_vector].col = j;
                    counter_vector++;
                }
            }
        }
    }

    ret=(struct core_adaptive_personalizzato *)malloc(sizeof(core_adaptive_personalizzato));
    if(ret==NULL) return  NULL;

    number_of_blocks_per_Grid+=( (total_row_scalar * 1 * K) + threadsPerBlock- 1) / threadsPerBlock;
    total_block_scalar=number_of_blocks_per_Grid;
    
    number_of_blocks_per_Grid+=( (total_row_vector * SUB_WARP_SIZE * K) + threadsPerBlock- 1) / threadsPerBlock;
    total_block_vector=number_of_blocks_per_Grid-total_block_scalar;

    metadata=(long *)malloc(sizeof(long) * 6);
    if(metadata==NULL) return NULL;

    /* Numero totale di blocchi per Griglia */
    metadata[0]=number_of_blocks_per_Grid;

    /* Numero di threads che devono computare gli elementi con CSR SCALAR */
    metadata[1]=total_row_scalar*K*1;

    /* Numero di thread che devono computare gli elementi con VECTOR SUB WARP */
    metadata[2]=total_row_vector*K*SUB_WARP_SIZE;

    /* Numero totale di blocchi per CSR SCALAR da utilizzare come threshold nel kernel */
    metadata[3]=total_block_scalar;

    /* Numero totale di blocchi per VECTOR SUB WARP */
    metadata[4]=total_block_vector;

    /* Numero totale di threads che dovranno computare gli elementi della matrice Y */
    metadata[5]=total_threads;

    printf("Numero totale dei blocchi nella griglia: %ld\n", metadata[0]);
    printf("Numero totale dei threads che devono computare gli elementi con CSR Scalar: %ld\n", metadata[1]);
    printf("Numero totale dei threads che devono computare gli elementi con VECTOR SUB WARP: %ld\n", metadata[2]);
    printf("Numero totale dei blocchi per CSR SCALAR: %ld\n", metadata[3]);
    printf("Numero totale dei blocchi per VECTOR SUB WARP: %ld\n", metadata[4]);
    printf("Numero totale dei threads che devono computare gli elementi di Y: %ld\n", metadata[5]);
    printf("Numero totale delle righe sopra la threshold %d: %ld\n", THR, total_row_vector);
    printf("Numero totale delle righe sotto la threshold %d: %ld\n", THR, total_row_scalar);

    ret->metadata=metadata;
    ret->items_scalar=items_scalar;
    ret->items_vector=items_vector;

    return ret;
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

double *CSR_GPU(int M, int N, int K, int nz, double *h_as, int *h_ja, int *h_irp, double **X, int *nz_per_row)
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

#ifdef CSR_ADAPTIVE_PERSONALIZZATO
    long *d_metadata = NULL;
    struct item* d_items_scalar=NULL;
    struct item* d_items_vector=NULL;
#endif

#if defined(CSR_VECTOR) || defined(CSR_VECTOR_BY_ROW)
    h_X = transpose_from_2D(N, K, X);
#else
    /* 2D to 1D dense matrix X conversion*/
    h_X = convert_2D_to_1D(N, K, X);
#endif

    /* Y array host memory allocation */
    memory_allocation(double, M *K, h_y);

    if (cudaDeviceReset() != cudaSuccess) {
		printf("cudaDeviceReset failed!\n");
		exit(1);
	}

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

#ifdef CSR_ADAPTIVE_PERSONALIZZATO

    struct core_adaptive_personalizzato *ret = csr_adaptive_personalizzato_number_of_blocks(M, nz_per_row, threadsPerBlock, K);

    /* Alloco e copio la struttura dati contenente i metadati */
    memory_allocation_Cuda(long, 6, d_metadata);    
    memcpy_to_dev(ret->metadata, d_metadata, long, 6);

    /* Alloco e copio la struttura dati contenente gli elementi della matrice Y che dovranno essere computati da CSR SCALAR */
    memory_allocation_Cuda(struct item, ret->metadata[1], d_items_scalar);    
    memcpy_to_dev(ret->items_scalar, d_items_scalar, struct item, ret->metadata[1]);

    /* Alloco e copio la struttura dati contenente gli elementi della matrice Y che dovranno essere computati da VECTOR SUB WARP */
    memory_allocation_Cuda(struct item, ret->metadata[2], d_items_vector);    
    memcpy_to_dev(ret->items_vector, d_items_vector, struct item, ret->metadata[2]);

    /* Number of blocks per grid */
    int blocksPerGrid=ret->metadata[0];

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

#if CSR_ADAPTIVE_PERSONALIZZATO

    CSR_Adaptive_personalizzato<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, nz, d_as, d_ja, d_irp, d_X, d_y, d_metadata, d_items_scalar, d_items_vector);

#elif CSR_VECTOR

    /* CSR Vector */
    CSR_Vector<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, nz, d_as, d_ja, d_irp, d_X, d_y);

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

    // STOP TIMER
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));

    err = cudaGetLastError();
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CSR kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

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
    
#ifdef CSR_ADAPTIVE_PERSONALIZZATO
	free_memory_Cuda(d_metadata);
	free_memory_Cuda(d_items_scalar);
	free_memory_Cuda(d_items_vector);
#endif

    /* Start the memory cleaning process on Host */

    printf("Freeing host memory ...\n");

    free(h_X);
    
#ifdef CSR_ADAPTIVE_PERSONALIZZATO
	free(ret->metadata);
	free(ret->items_scalar);
	free(ret->items_vector);
#endif

    printf("Completed parallel product CSR ...\n");

    return h_y;
}
