#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda_runtime.h> // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h> // For CUDA SDK timers

#include "../include/header.h"

#ifdef CSR
#define csr_scalar 0
#define csr_vector 1
#define csr_vector_sub_warp 2
#define csr_adaptive 3

void samplings_GPU_CSR(int M, int N, int nz, double *h_as, int *h_ja, int *h_irp)
{
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;

    int K[] = {1, 3, 4, 8, 12, 16, 32, 64};

    int modes[] = {csr_scalar, csr_vector,csr_vector_sub_warp, csr_adaptive};

    FILE *f_samplings;
    /**
     * Name of the file to be created and written to
     */
    const char *fn;

    // HOST
    double *h_y = NULL;
    double *h_X = NULL;

    // DEVICE
    double *d_y = NULL;
    double *d_X = NULL;
    double *d_as = NULL;
    int *d_ja = NULL;
    int *d_irp = NULL;

    // SAMPLINGS variables
    double M2 = 0.0;
    double variance = 0.0;
    double Gflops = 0.0;
    double curr_Gflops = 0.0;

    float expireTimeMsec = 0.0;

    // CSR ADAPTIVE variables
    int *rowBlocks = NULL;
    int *d_rowBlocks = NULL;

    /* Number of threads per block */
    int threadsPerBlock = MAX_BLOCK_DIM;
    /* Number of blocks per grid */
    int blocksPerGrid;
    /* Number of elements in the resulting matrix y */
    int numElements;
    /*Number of blocks to be spawned in the adaptive algorithm */
    int number_of_blocks;
    /* Number of warps per block*/
    int warpsPerBlock;
    /* sub_warp_size is the power of 2 closest to the mean (rounded down) of non-zeros per row*/
    int sub_warp_size = pow(2, floor(log2((nz + M - 1) / M)));
    if (sub_warp_size > WARP_SIZE) sub_warp_size = WARP_SIZE;

    printf("Allocating device variables for CPU CSR product ...\n");

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

    /**
     * Opening the output file
     */
    fn = "plots/samplings_CSR_GPU.csv";
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "Algorithm,K,GFLOPS,GFLOPS_variability\n");

    // Creating cuda events
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    for (int k = 0; k < sizeof(K) / sizeof(int); k++)
    {
          /**
         * Creating the X matrix with its number of columns specified by K[k]
        */
        create_dense_matrix_1D(N, K[k], &h_X);

        /* Y array host memory allocation */
        memory_allocation(double, M *K[k], h_y);

        /* Y array host memory allocation */
        memory_allocation_Cuda(double, M *K[k], d_y);
        /* Device allocation for dense matrix X */
        memory_allocation_Cuda(double, N *K[k], d_X);

        memcpy_to_dev(h_X, d_X, double, N *K[k]);

         /* Number of elements of the product matrix Y */
        numElements = M * K[k];

        for (int i = 0; i < sizeof(modes) / sizeof(int); i++)
        {

            switch (i)
            {
            case csr_adaptive:

                number_of_blocks = csr_adaptive_rowblocks(M, nz, h_irp, &rowBlocks, &threadsPerBlock);

                /* Device allocation for d_rowBlocks */
                memory_allocation_Cuda(int, number_of_blocks, d_rowBlocks);

                /* Copy rowBlocks from the Host to the Device*/
                memcpy_to_dev(rowBlocks, d_rowBlocks, int, number_of_blocks);
                /* Number of blocks per grid */
                blocksPerGrid = (number_of_blocks - 1) * K[k];
                break;

            case csr_scalar:

                threadsPerBlock = MAX_BLOCK_DIM;

                /* Number of blocks per grid */
                blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
                break;
            case csr_vector:

                threadsPerBlock = MAX_BLOCK_DIM;

                warpsPerBlock = threadsPerBlock / WARP_SIZE;

                /* Number of blocks per grid */
                blocksPerGrid = (numElements + warpsPerBlock - 1) / warpsPerBlock;
                break;
            case csr_vector_sub_warp:

                threadsPerBlock = MAX_BLOCK_DIM;

                /* Number of blocks per grid */
                blocksPerGrid = (numElements +  threadsPerBlock / sub_warp_size - 1) / (threadsPerBlock / sub_warp_size) ;
                break;
            default:
                printf("The mode is invalid\n");
                exit(1);
                break;
            }

             /**
             * Resetting the average stats
            */

            M2 = 0.0;
            variance = 0.0;
            Gflops = 0.0;
            curr_Gflops = 0.0;
            for (int curr_samp = 0; curr_samp < SAMPLING_SIZE; curr_samp++)
            {
                printf("CUDA kernel for K = %d launch with %d blocks of %d threads\n", K[k], blocksPerGrid,
                       threadsPerBlock);

                // START TIMER
                checkCudaErrors(cudaEventRecord(start, stream));

                switch (i)
                {
                case csr_adaptive:
                    /* CSR Adaptive */
                    CSR_Adaptive_Kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y, d_rowBlocks);
                    break;
                case csr_vector:
                    /* CSR Vector */
                    CSR_Vector_Kernel<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y);
                    break;
                case csr_vector_sub_warp:
                    /* CSR Vector with sub-warps*/
                    CSR_Vector_Sub_warp<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y, sub_warp_size);
                    break;
                case csr_scalar:

                    /* Versione accesso alla memoria globale non ottimizzato */
                    // CSR_kernel_v1<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y, numElements);

                    // CSR_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y, numElements);

                    /* Versione accesso alla memoria globale ottimizzato */
                    CSR_kernel_v3<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y, numElements);
                    break;

                default:
                    printf("The mode is invalid\n");
                    exit(1);
                    break;
                }

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

                /**
                * Welford's one-pass algorithm is an efficient method for computing mean and variance in a single pass over a sequence of values.
                * It achieves this by updating the mean and variance incrementally as new values are encountered.
                */

                curr_Gflops = compute_GFLOPS(K[k], nz, expireTimeMsec * 1e6);
                Gflops = calculate_mean(curr_Gflops, Gflops, curr_samp + 1);
                M2 = calculate_M2(curr_Gflops, Gflops, M2, curr_samp + 1);
            }
            /*After processing all values, the variance can be calculated as M2 / (n - 1).*/
            variance = M2 / (SAMPLING_SIZE - 1);

            printf("GLOPS MEAN (GLOPS VARIANCE %lf) FOR PARALLEL PRODUCT GPU with K = %d is: %lf\n", variance, K[k], Gflops);

             /**
             * Writing in the file the GLOPS mean and variance
            */

            switch (i)
            {
            case csr_adaptive:
                fprintf(f_samplings, "csr_adaptive,%d, %lf,%.20lf\n", K[k], Gflops, variance);
                free_memory_Cuda(d_rowBlocks);
                free(rowBlocks);
                break;
            case csr_vector:
                fprintf(f_samplings, "csr_vector,%d, %lf,%.20lf\n", K[k], Gflops, variance);
                break;
            case csr_vector_sub_warp:
                fprintf(f_samplings, "csr_vector_sub_warp,%d, %lf,%.20lf\n", K[k], Gflops, variance);
                break;
            case csr_scalar:
                fprintf(f_samplings, "csr_scalar,%d, %lf,%.20lf\n", K[k], Gflops, variance);
                break;

            default:
                printf("The mode is invalid\n");
                exit(1);
                break;
            }

            fflush(f_samplings);
        }

        free_memory_Cuda(d_X);
        free_memory_Cuda(d_y);
        free(h_X);
        free(h_y);
    }

    /* Start the memory cleaning process on Device */
    printf("Freeing Device memory ...\n");
    if (h_as != NULL)
        free_memory_Cuda(d_as);
    free_memory_Cuda(d_ja);

    /**
     * Closing the file
    */

    if (f_samplings != stdin)
        fclose(f_samplings);

    return;
}
#elif ELLPACK

#define ellpack 0
#define ellpack_sub_warp 1

void samplings_GPU_ELLPACK(int M, int N, int nz, int *nz_per_row, double **values, int **col_indices)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;

    int K[] = {1, 3, 4, 8, 12, 16, 32, 64};

    int modes[] = {ellpack, ellpack_sub_warp};

    FILE *f_samplings;
    /**
     * Name of the file to be created and written to
     */
    const char *fn;

    // host variables
    double *h_y = NULL;
    double *h_X = NULL;
    double *h_values = NULL;
    int *h_col_indices = NULL;
    int *h_sum_nz = NULL;

    //Device variables
    double *d_X = NULL;
    double *d_y = NULL;
    double *d_values = NULL;
    int *d_col_indices = NULL;

    int *d_nz_per_row = NULL;
    int *d_sum_nz = NULL;

    // SAMPLINGS variables
    double M2 = 0.0;
    double variance = 0.0;
    double Gflops = 0.0;
    double curr_Gflops = 0.0;

    /* Number of threads per block */
    int threadsPerBlock = MAX_BLOCK_DIM;
    /** Number of blocks per grid*/
    int blocksPerGrid;
    /*Total number of elements to be computed*/
    int numElements;

    /*sub_warp_size is the power of 2 closest to the mean (rounded down) of non-zeros per row*/
    int sub_warp_size = pow(2, floor(log2((nz + M - 1) / M)));
    if (sub_warp_size > WARP_SIZE)
        sub_warp_size = WARP_SIZE;

    /*Number of warps per block*/    
    int warpsPerBlock = threadsPerBlock / sub_warp_size;

    float expireTimeMsec = 0.0;

    /* irregular 2D to 1D conversions*/
    if (values != NULL)
        h_values = convert_2D_to_1D_per_ragged_matrix(M, nz, nz_per_row, values);
    h_col_indices = convert_2D_to_1D_per_ragged_matrix(M, nz, nz_per_row, col_indices);

    printf("Allocating device variables for CPU ELLPACK product ...\n");

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
    /* Copy of the contents of nz_per_row from the Host to the Device */
    memcpy_to_dev(nz_per_row, d_nz_per_row, int, M);

    h_sum_nz = compute_sum_nz(M, nz_per_row);
    /* Copy of the contents of h_sum_nz from the Host to the Device */
    memcpy_to_dev(h_sum_nz, d_sum_nz, int, M);

    /**
     * Opening the output file
     */
    fn = "plots/samplings_ELLPACK_GPU.csv";
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "Algorithm,K,GFLOPS,GFLOPS_variability\n");

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    for (int k = 0; k < sizeof(K) / sizeof(int); k++)
    {
        /**
         * Creating the X matrix with its number of columns specified by K[k]
        */
        create_dense_matrix_1D(N, K[k], &h_X);

        /* Y array host memory allocation */
        memory_allocation(double, M *K[k], h_y);
        /* Y array host memory allocation */
        memory_allocation_Cuda(double, M *K[k], d_y);
        /* Device allocation for dense matrix X */
        memory_allocation_Cuda(double, N *K[k], d_X);

        /* Copy of the contents of the dense vector h_X from the Host to the Devicee */
        memcpy_to_dev(h_X, d_X, double, N *K[k]);

        /* Number of elements of the product matrix Y */
        numElements = M * K[k];

        for (int i = 0; i < sizeof(modes) / sizeof(int); i++)
        {
            switch (i)
            {
            case ellpack:
                /* Number of blocks per grid */
                blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
                break;
            case ellpack_sub_warp:
                /* Number of blocks per grid */
                blocksPerGrid = (numElements + warpsPerBlock - 1) / warpsPerBlock;
                break;
            default:
                printf("The mode is invalid\n");
                exit(1);
                break;
            }

             /**
             * Resetting the average stats
            */

            curr_Gflops = 0.0;
            M2 = 0.0;
            variance = 0.0;
            Gflops = 0.0;

            for (int curr_samp = 0; curr_samp < SAMPLING_SIZE; curr_samp++)
            {
                printf("CUDA kernel for K = %d launch with %d blocks of %d threads\n", K[k], blocksPerGrid,
                       threadsPerBlock);
                // START TIMER
                checkCudaErrors(cudaEventRecord(start, stream));
                switch (i)
                {
                case ellpack:
                    ELLPACK_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], d_nz_per_row, d_sum_nz, d_values, d_col_indices, d_X, d_y);
                    break;
                case ellpack_sub_warp:
                    ELLPACK_Sub_warp<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], d_nz_per_row, d_sum_nz, d_values, d_col_indices, d_X, d_y, sub_warp_size);
                    break;
                default:
                    printf("The mode is invalid\n");
                    exit(1);
                    break;
                }

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
                
                /**
                * Welford's one-pass algorithm is an efficient method for computing mean and variance in a single pass over a sequence of values.
                * It achieves this by updating the mean and variance incrementally as new values are encountered.
                */

                curr_Gflops = compute_GFLOPS(K[k], nz, expireTimeMsec * 1e6);
                Gflops = calculate_mean(curr_Gflops, Gflops, curr_samp + 1);
                M2 = calculate_M2(curr_Gflops, Gflops, M2, curr_samp + 1);
            }
            /*After processing all values, the variance can be calculated as M2 / (n - 1).*/
            variance = M2 / (SAMPLING_SIZE - 1);

            printf("GLOPS MEAN (GLOPS VARIANCE %lf) FOR PARALLEL PRODUCT GPU with K = %d is: %lf\n", variance, K[k], Gflops);

             /**
             * Writing in the file the GLOPS mean and variance
            */

            switch (i)
            {
            case ellpack:
                fprintf(f_samplings, "ellpack,%d, %lf,%.20lf\n", K[k], Gflops, variance);
                break;
            case ellpack_sub_warp:
                fprintf(f_samplings, "ellpack_sub_warp,%d, %lf,%.20lf\n", K[k], Gflops, variance);
                break;
            default:
                printf("The mode is invalid\n");
                exit(1);
                break;
            }

            fflush(f_samplings);
        }

        free_memory_Cuda(d_X);
        free_memory_Cuda(d_y);
        if (h_X != NULL)
            free(h_X);
        if (h_y != NULL)
            free(h_y);
    }

    /* Start the memory cleaning process on Device */
    printf("Freeing Device memory ...\n");
    if (values != NULL)
        free_memory_Cuda(d_values);
    free_memory_Cuda(d_col_indices);
    free_memory_Cuda(d_nz_per_row);
    free_memory_Cuda(d_sum_nz);

    /* Start the memory cleaning process on Host */
    printf("Freeing host memory ...\n");

    if (h_values != NULL)
        free(h_values);
    if (h_col_indices != NULL)
        free(h_col_indices);
    if (h_sum_nz != NULL)
        free(h_sum_nz);

     /**
     * Closing the file
    */
    if (f_samplings != stdin)
        fclose(f_samplings);

}

#endif