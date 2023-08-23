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
#define csr_vector_by_row 3
#define csr_adaptive_personalizzato 4

/**
 *
 * samplings_GPU_CSR - Function that computs for a fixed number of samplings, the mean and the variance of the GFOLPS as the algorithm and K vary.
 * The overall results are written in a proper .csv file.
 *
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param h_as: Coefficient vector
 * @param h_ja: Column index vector
 * @param h_irp: Vector of the start index of each row

 */
void samplings_GPU_CSR(int M, int N, int nz, double *h_as, int *h_ja, int *h_irp, int *nz_per_row)
{
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;

    int K[] = {1, 3, 4, 8, 12, 16, 32, 64};

    int modes[] = {csr_scalar, csr_vector, csr_vector_sub_warp, csr_vector_by_row, csr_adaptive_personalizzato};

    FILE *f_samplings;
    /**
     * Name of the file to be created and written to
     */
    char fn[100];

    // HOST
    double *h_y = NULL;
    double *h_X = NULL;
    double *h_X_t = NULL;

    // DEVICE
    double *d_y = NULL;
    double *d_X = NULL;
    double *d_X_t = NULL;
    double *d_as = NULL;
    int *d_ja = NULL;
    int *d_irp = NULL;

    // SAMPLINGS variables
    double M2 = 0.0;
    double variance = 0.0;
    double Gflops = 0.0;
    double curr_Gflops = 0.0;

    float expireTimeMsec = 0.0;

    /* Number of threads per block */
    int threadsPerBlock = MAX_BLOCK_DIM;
    /* Number of blocks per grid */
    int blocksPerGrid;
    /* Number of elements in the resulting matrix y */
    int numElements;
    /* Number of warps per block*/
    int warpsPerBlock;
    /* Number of sub-warps per block*/
    int subWarpsPerBlock;

    int sub_warp_size = 2;

    printf("Allocating device variables for CPU CSR product ...\n");

	// CSR ADAPTIVE PERSONALIZZATO
    long *d_metadata = NULL;
    struct item* d_items_scalar=NULL;
    struct item* d_items_vector=NULL;
    struct core_adaptive_personalizzato *ret;

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
    /* Copy of the contents of the vector irp from the Host to the Devicee */
    memcpy_to_dev(h_irp, d_irp, int, M + 1);
    /* Copy of the dense vector X from the Host to the Device*/

    /**
     * Opening the output file
     */
    printf("Opening the output file\n");

    char *token;
    token = strtok(filename, "/");
    token = strtok(NULL, "/");

    sprintf(fn, "plots/samplings_CSR_GPU_%s.csv", token);
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "Algorithm,K,GFLOPS,GFLOPS_variability\n");

    // Creating cuda events
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    for (int k = 0; k < sizeof(K) / sizeof(int); k++)
    {

        /* Y array host memory allocation */
        memory_allocation(double, M *K[k], h_y);

        /* Y array host memory allocation */
        memory_allocation_Cuda(double, M *K[k], d_y);
        
        /* The output matrix is initialized with all zeroes */
        cudaMemset(d_y, 0, M * K[k] * sizeof(double));

        /* Creating the X matrix with its number of columns specified by K[k] */

        create_dense_matrix_1D(N, K[k], &h_X);
        
        h_X_t = transpose_from_1D(N, K[k], h_X);

        /* Device allocation for dense matrix X */
        memory_allocation_Cuda(double, N *K[k], d_X);

        memcpy_to_dev(h_X, d_X, double, N *K[k]);
        
        /* Device allocation for dense matrix X */
        memory_allocation_Cuda(double, N *K[k], d_X_t);

        memcpy_to_dev(h_X_t, d_X_t, double, N *K[k]);

        /* Number of elements of the product matrix Y */
        numElements = M * K[k];

        for (int i = 0; i < sizeof(modes) / sizeof(int); i++)
        {

            switch (i)
            {
            case csr_adaptive_personalizzato:

                ret = csr_adaptive_personalizzato_number_of_blocks(M, nz_per_row, threadsPerBlock, K[k]);
		
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
                blocksPerGrid=ret->metadata[0];
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

            case csr_vector_by_row:

                threadsPerBlock = MAX_BLOCK_DIM;

                warpsPerBlock = threadsPerBlock / WARP_SIZE;

                /* Number of blocks per grid */
                blocksPerGrid = (M + warpsPerBlock - 1) / warpsPerBlock;
                break;

            case csr_vector_sub_warp:

                threadsPerBlock = MAX_BLOCK_DIM;

                subWarpsPerBlock = threadsPerBlock / sub_warp_size;

                /* Number of blocks per grid */
                blocksPerGrid = (numElements + subWarpsPerBlock - 1) / subWarpsPerBlock;
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
		
                printf("CUDA kernel for K = %d launch with %d blocks of %d threads\n", K[k], blocksPerGrid, threadsPerBlock);
                fflush(stdout);
                
                // START TIMER
                checkCudaErrors(cudaEventRecord(start, stream));

                switch (i)
                {
                case csr_adaptive_personalizzato:
                	/* CSR Adaptive personalizzato */
                    CSR_Adaptive_personalizzato<<<blocksPerGrid, threadsPerBlock>>>(M, N, K[k], nz, d_as, d_ja, d_irp, d_X, d_y, d_metadata, d_items_scalar, d_items_vector);
                    break;

                case csr_vector:
                    /* CSR Vector */
                    CSR_Vector<<<blocksPerGrid, threadsPerBlock>>>(M, N, K[k], nz, d_as, d_ja, d_irp, d_X_t, d_y);
                    break;
                    
                case csr_vector_by_row:
					/* CSR Vector By Row */
                    CSR_Vector_by_row<<<blocksPerGrid, threadsPerBlock>>>(M, N, K[k], nz, d_as, d_ja, d_irp, d_X_t, d_y);
                    break;

                case csr_vector_sub_warp:
                    /* CSR Vector with sub-warps*/
                    CSR_Vector_Sub_warp<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y, sub_warp_size);
                    break;

                case csr_scalar:
					/* CSR Scalar */
                    /* Versione accesso alla memoria globale non ottimizzato */
                    // CSR_Scalar_v1<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y);
                    // CSR_Scalar_v2<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y);
                    /* Versione accesso alla memoria globale ottimizzato */
                    CSR_Scalar_v3<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y);
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

                M2 = calculate_M2(curr_Gflops, Gflops, M2, curr_samp + 1);

                Gflops = calculate_mean(curr_Gflops, Gflops, curr_samp + 1);

            }
            /*After processing all values, the variance can be calculated as M2 / (n).*/
            variance = M2 / (SAMPLING_SIZE);

            printf("GLOPS MEAN (GLOPS VARIANCE %lf) FOR PARALLEL PRODUCT GPU with K = %d is: %lf\n", variance, K[k], Gflops);

            /**
             * Writing in the file the GLOPS mean and variance
             */

            switch (i)
            {
            
            case csr_adaptive_personalizzato:
                fprintf(f_samplings, "csr_adaptive_personalizzato,%d, %lf,%.20lf\n", K[k], Gflops, variance);
                free_memory_Cuda(d_metadata);
                free_memory_Cuda(d_items_scalar);
                free_memory_Cuda(d_items_vector);
                free(ret->metadata);
    			free(ret->items_scalar);
    			free(ret->items_vector);
                free(ret);
                break;

            case csr_vector:
                fprintf(f_samplings, "csr_vector,%d, %lf,%.20lf\n", K[k], Gflops, variance);
                break;
                
            case csr_vector_by_row:
                fprintf(f_samplings, "csr_vector_by_row,%d, %lf,%.20lf\n", K[k], Gflops, variance);
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
        free_memory_Cuda(d_X_t);
        free_memory_Cuda(d_y);
        free(h_X);
        free(h_X_t);
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


/**
 *
 * samplings_GPU_CSR - Function that computs for a fixed number of samplings, the mean and the variance of the GFOLPS as the algorithm and K vary.
 * The overall results are written in a proper .csv file.
 *
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param h_as: Coefficient vector
 * @param h_ja: Column index vector
 * @param h_irp: Vector of the start index of each row

 */
void samplings_GPU_CSR_flush_cache(int M, int N, int nz, double *h_as, int *h_ja, int *h_irp, int *nz_per_row)
{
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;

    int K[] = {1, 3, 4, 8, 12, 16, 32, 64};

    int modes[] = {csr_scalar, csr_vector, csr_vector_sub_warp, csr_vector_by_row, csr_adaptive_personalizzato};

    FILE *f_samplings;
    /**
     * Name of the file to be created and written to
     */
    char fn[100];

    // HOST
    double *h_y = NULL;
    double *h_X = NULL;
    double *h_X_t = NULL;

    // DEVICE
    double *d_y = NULL;
    double *d_X = NULL;
    double *d_X_t = NULL;
    double *d_as = NULL;
    int *d_ja = NULL;
    int *d_irp = NULL;

    // SAMPLINGS variables
    double M2 = 0.0;
    double variance = 0.0;
    double Gflops = 0.0;
    double curr_Gflops = 0.0;

    float expireTimeMsec = 0.0;

    /* Number of threads per block */
    int threadsPerBlock = MAX_BLOCK_DIM;
    /* Number of blocks per grid */
    int blocksPerGrid;
    /* Number of elements in the resulting matrix y */
    int numElements;
    /* Number of warps per block*/
    int warpsPerBlock;
    /* Number of sub-warps per block*/
    int subWarpsPerBlock;

    int sub_warp_size = 2;

    

	// CSR ADAPTIVE PERSONALIZZATO
    long *d_metadata = NULL;
    struct item* d_items_scalar=NULL;
    struct item* d_items_vector=NULL;
    struct core_adaptive_personalizzato *ret;

    /**
     * Opening the output file
     */
    printf("Opening the output file\n");

    char *token;
    token = strtok(filename, "/");
    token = strtok(NULL, "/");

    sprintf(fn, "plots/samplings_CSR_GPU_%s.csv", token);
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "Algorithm,K,GFLOPS,GFLOPS_variability\n");



    for (int k = 0; k < sizeof(K) / sizeof(int); k++)
    {
        /* Creating the X matrix with its number of columns specified by K[k] */
        create_dense_matrix_1D(N, K[k], &h_X);
        
        h_X_t = transpose_from_1D(N, K[k], h_X);

        /* Number of elements of the product matrix Y */
        numElements = M * K[k];

        for (int i = 0; i < sizeof(modes) / sizeof(int); i++)
        {

            switch (i)
            {
            case csr_adaptive_personalizzato:

                ret = csr_adaptive_personalizzato_number_of_blocks(M, nz_per_row, threadsPerBlock, K[k]);

                /* Number of blocks per grid */
                blocksPerGrid=ret->metadata[0];
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

            case csr_vector_by_row:

                threadsPerBlock = MAX_BLOCK_DIM;

                warpsPerBlock = threadsPerBlock / WARP_SIZE;

                /* Number of blocks per grid */
                blocksPerGrid = (M + warpsPerBlock - 1) / warpsPerBlock;
                break;

            case csr_vector_sub_warp:

                threadsPerBlock = MAX_BLOCK_DIM;

                subWarpsPerBlock = threadsPerBlock / sub_warp_size;

                /* Number of blocks per grid */
                blocksPerGrid = (numElements + subWarpsPerBlock - 1) / subWarpsPerBlock;
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
            	
				if (cudaDeviceReset() != cudaSuccess) {
					printf("cudaDeviceReset failed!\n");
					exit(1);
				}
				
				printf("Allocating device variables for CPU CSR product ...\n");
		
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
				/* Copy of the contents of the vector irp from the Host to the Devicee */
				memcpy_to_dev(h_irp, d_irp, int, M + 1);
				
				
				/* Y array host memory allocation */
				memory_allocation(double, M *K[k], h_y);

				/* Y array host memory allocation */
				memory_allocation_Cuda(double, M *K[k], d_y);
				
				/* The output matrix is initialized with all zeroes */
				cudaMemset(d_y, 0, M * K[k] * sizeof(double));
				
				/* Device allocation for dense matrix X */
				memory_allocation_Cuda(double, N *K[k], d_X);

				memcpy_to_dev(h_X, d_X, double, N *K[k]);
				
				/* Device allocation for dense matrix X */
				memory_allocation_Cuda(double, N *K[k], d_X_t);

				memcpy_to_dev(h_X_t, d_X_t, double, N *K[k]);		
		
                printf("CUDA kernel for K = %d launch with %d blocks of %d threads\n", K[k], blocksPerGrid, threadsPerBlock);
                
                fflush(stdout);
                
				// Creating cuda events
				checkCudaErrors(cudaEventCreate(&start));
				checkCudaErrors(cudaEventCreate(&stop));

                switch (i)
                {
                
                case csr_adaptive_personalizzato:
                
					/* Alloco e copio la struttura dati contenente i metadati */
		            memory_allocation_Cuda(long, 6, d_metadata);    
		            memcpy_to_dev(ret->metadata, d_metadata, long, 6);

		            /* Alloco e copio la struttura dati contenente gli elementi della matrice Y che dovranno essere computati da CSR SCALAR */
		            memory_allocation_Cuda(struct item, ret->metadata[1], d_items_scalar);    
		            memcpy_to_dev(ret->items_scalar, d_items_scalar, struct item, ret->metadata[1]);

		            /* Alloco e copio la struttura dati contenente gli elementi della matrice Y che dovranno essere computati da VECTOR SUB WARP */
		            memory_allocation_Cuda(struct item, ret->metadata[2], d_items_vector);    
		            memcpy_to_dev(ret->items_vector, d_items_vector, struct item, ret->metadata[2]);
                
                	// START TIMER
                	checkCudaErrors(cudaEventRecord(start, stream));
                	
                	/* CSR Adaptive personalizzato */
                    CSR_Adaptive_personalizzato<<<blocksPerGrid, threadsPerBlock>>>(M, N, K[k], nz, d_as, d_ja, d_irp, d_X, d_y, d_metadata, d_items_scalar, d_items_vector);
                    
		            // STOP TIMER
		            checkCudaErrors(cudaEventRecord(stop, stream));
		            checkCudaErrors(cudaEventSynchronize(stop));
		            checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));
                    
		            
		            free_memory_Cuda(d_metadata);
		            free_memory_Cuda(d_items_scalar);
		            free_memory_Cuda(d_items_vector);
                    break;

                case csr_vector:
                    /* CSR Vector */
                    // START TIMER
                	checkCudaErrors(cudaEventRecord(start, stream));

                    CSR_Vector<<<blocksPerGrid, threadsPerBlock>>>(M, N, K[k], nz, d_as, d_ja, d_irp, d_X_t, d_y);
		            // STOP TIMER
		            checkCudaErrors(cudaEventRecord(stop, stream));
		            checkCudaErrors(cudaEventSynchronize(stop));
		            checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));
                    break;
                    
                case csr_vector_by_row:
					/* CSR Vector By Row */
					// START TIMER
                	checkCudaErrors(cudaEventRecord(start, stream));

                    CSR_Vector_by_row<<<blocksPerGrid, threadsPerBlock>>>(M, N, K[k], nz, d_as, d_ja, d_irp, d_X_t, d_y);
		            // STOP TIMER
		            checkCudaErrors(cudaEventRecord(stop, stream));
		            checkCudaErrors(cudaEventSynchronize(stop));
		            checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));
                    break;

                case csr_vector_sub_warp:
                    /* CSR Vector with sub-warps*/
					// START TIMER
                	checkCudaErrors(cudaEventRecord(start, stream));

                    CSR_Vector_Sub_warp<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y, sub_warp_size);
		            // STOP TIMER
		            checkCudaErrors(cudaEventRecord(stop, stream));
		            checkCudaErrors(cudaEventSynchronize(stop));
		            checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));
                    break;

                case csr_scalar:
					/* CSR Scalar */
                    /* Versione accesso alla memoria globale non ottimizzato */
                    // CSR_Scalar_v1<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y);
                    // CSR_Scalar_v2<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y);
                    /* Versione accesso alla memoria globale ottimizzato */

					// START TIMER
                	checkCudaErrors(cudaEventRecord(start, stream));

                    CSR_Scalar_v3<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], nz, d_as, d_ja, d_irp, d_X, d_y);
		            // STOP TIMER
		            checkCudaErrors(cudaEventRecord(stop, stream));
		            checkCudaErrors(cudaEventSynchronize(stop));
		            checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));
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

                free_memory_Cuda(d_X);
				free_memory_Cuda(d_X_t);
				free_memory_Cuda(d_y);

                /**
                 * Welford's one-pass algorithm is an efficient method for computing mean and variance in a single pass over a sequence of values.
                 * It achieves this by updating the mean and variance incrementally as new values are encountered.
                 */
                    
                curr_Gflops = compute_GFLOPS(K[k], nz, expireTimeMsec * 1e6);

                M2 = calculate_M2(curr_Gflops, Gflops, M2, curr_samp + 1);

                Gflops = calculate_mean(curr_Gflops, Gflops, curr_samp + 1);

            }
            /*After processing all values, the variance can be calculated as M2 / (n).*/
            variance = M2 / (SAMPLING_SIZE);

            printf("GLOPS MEAN (GLOPS VARIANCE %lf) FOR PARALLEL PRODUCT GPU with K = %d is: %lf\n", variance, K[k], Gflops);

            /**
             * Writing in the file the GLOPS mean and variance
             */

            switch (i)
            {
                
            case csr_adaptive_personalizzato:
				fprintf(f_samplings, "csr_adaptive_personalizzato,%d, %lf,%.20lf\n", K[k], Gflops, variance);
				free(ret->metadata);
				free(ret->items_scalar);
				free(ret->items_vector);
		        free(ret);
                break;

            case csr_vector:
                fprintf(f_samplings, "csr_vector,%d, %lf,%.20lf\n", K[k], Gflops, variance);
                break;
                
            case csr_vector_by_row:
                fprintf(f_samplings, "csr_vector_by_row,%d, %lf,%.20lf\n", K[k], Gflops, variance);
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

        free(h_X);
        free(h_X_t);
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

/**
 *
 * samplings_GPU_ELLPACK - Function that computs for a fixed number of samplings, the mean and the variance of the GFOLPS as the algorithm and K vary.
 * The overall results are written in a proper .csv file.
 *
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param nz_per_row: Array containing the number of non-zeroes per row
 * @param values: 2D array of coefficients
 * @param col_indices: 2D array of column indexes

 */

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
    char fn[100];

    // host variables
    double *h_y = NULL;
    double *h_X = NULL;
    double *h_values = NULL;
    int *h_col_indices = NULL;
    int *h_sum_nz = NULL;

    // Device variables
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

    int sub_warp_size = 2;

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
    printf("Opening the output file\n");

    char *token;
    token = strtok(filename, "/");
    token = strtok(NULL, "/");

    sprintf(fn, "plots/samplings_ELLPACK_GPU_%s.csv", token);
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

                M2 = calculate_M2(curr_Gflops, Gflops, M2, curr_samp + 1);

                Gflops = calculate_mean(curr_Gflops, Gflops, curr_samp + 1);

            }
            /*After processing all values, the variance can be calculated as M2 / (n).*/
            variance = M2 / (SAMPLING_SIZE);

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


/**
*
* samplings_GPU_ELLPACK_flush_cache - Function that computs for a fixed number of samplings, the mean and the variance of the GFOLPS as the algorithm and K vary.
* The overall results are written in a proper .csv file. The cache is properly flushed
*
* @param M: Number of rows
* @param N: Number of columns
* @param nz: Number of nz
* @param nz_per_row: Array containing the number of non-zeroes per row
* @param values: 2D array of coefficients
* @param col_indices: 2D array of column indexes
*/
 

void samplings_GPU_ELLPACK_flush_cache(int M, int N, int nz, int *nz_per_row, double **values, int **col_indices)
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
    char fn[100]; 

    // host variables
    double *h_y = NULL;
    double *h_X = NULL;
    double *h_values = NULL;
    int *h_col_indices = NULL;
    int *h_sum_nz = NULL;

    // Device variables
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

    int sub_warp_size = 2;

    /*Number of warps per block*/
    int warpsPerBlock = threadsPerBlock / sub_warp_size;

    float expireTimeMsec = 0.0; 

    /* irregular 2D to 1D conversions*/
    if (values != NULL)
        h_values = convert_2D_to_1D_per_ragged_matrix(M, nz, nz_per_row, values);
        
    h_col_indices = convert_2D_to_1D_per_ragged_matrix(M, nz, nz_per_row, col_indices); 

    h_sum_nz = compute_sum_nz(M, nz_per_row);

     /* Opening the output file*/
    printf("Opening the output file\n"); 

    char *token;
    token = strtok(filename, "/");
    token = strtok(NULL, "/"); 

    sprintf(fn, "plots/samplings_ELLPACK_GPU_%s.csv", token);
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "Algorithm,K,GFLOPS,GFLOPS_variability\n"); 

    for (int k = 0; k < sizeof(K) / sizeof(int); k++)
    {
        /**
         * Creating the X matrix with its number of columns specified by K[k]
         */
        create_dense_matrix_1D(N, K[k], &h_X); 

        /* Y array host memory allocation */
        memory_allocation(double, M *K[k], h_y);

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

				if (cudaDeviceReset() != cudaSuccess) {
					printf("cudaDeviceReset failed!\n");
					exit(1);
				} 

                printf("Allocating device variables for CPU ELLPACK product ...\n");

                /* Y array host memory allocation */
                memory_allocation_Cuda(double, M *K[k], d_y);
                /* Device allocation for dense matrix X */
                memory_allocation_Cuda(double, N *K[k], d_X);

                /* Copy of the contents of the dense vector h_X from the Host to the Devicee */
                memcpy_to_dev(h_X, d_X, double, N *K[k]); 

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

                /* Copy of the contents of h_sum_nz from the Host to the Device */
                memcpy_to_dev(h_sum_nz, d_sum_nz, int, M); 

                checkCudaErrors(cudaEventCreate(&start));
                checkCudaErrors(cudaEventCreate(&stop));

                printf("CUDA kernel for K = %d launch with %d blocks of %d threads\n", K[k], blocksPerGrid, threadsPerBlock);

                switch (i)
                {
                case ellpack: 

                    // START TIMER
                    checkCudaErrors(cudaEventRecord(start, stream)); 

                    ELLPACK_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], d_nz_per_row, d_sum_nz, d_values, d_col_indices, d_X, d_y); 

                    // STOP TIMER
                    checkCudaErrors(cudaEventRecord(stop, stream));
                    checkCudaErrors(cudaEventSynchronize(stop));
                    checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));
                    break;
                case ellpack_sub_warp:

                    // START TIMER
                    checkCudaErrors(cudaEventRecord(start, stream));
                    ELLPACK_Sub_warp<<<blocksPerGrid, threadsPerBlock>>>(M, K[k], d_nz_per_row, d_sum_nz, d_values, d_col_indices, d_X, d_y, sub_warp_size); 

                    // STOP TIMER
                    checkCudaErrors(cudaEventRecord(stop, stream));
                    checkCudaErrors(cudaEventSynchronize(stop));
                    checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));
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

 

                /**
                 * Welford's one-pass algorithm is an efficient method for computing mean and variance in a single pass over a sequence of values.
                 * It achieves this by updating the mean and variance incrementally as new values are encountered.
                 */

 

                curr_Gflops = compute_GFLOPS(K[k], nz, expireTimeMsec * 1e6);

                M2 = calculate_M2(curr_Gflops, Gflops, M2, curr_samp + 1);

                Gflops = calculate_mean(curr_Gflops, Gflops, curr_samp + 1);

            }
            /*After processing all values, the variance can be calculated as M2 / (n).*/
            variance = M2 / (SAMPLING_SIZE);

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

            /* Start the memory cleaning process on Device */
            printf("Freeing Device memory ...\n"); 

            free_memory_Cuda(d_X);
            free_memory_Cuda(d_y); 

            if (values != NULL)
                free_memory_Cuda(d_values);
                
            free_memory_Cuda(d_col_indices);
            free_memory_Cuda(d_nz_per_row);
            free_memory_Cuda(d_sum_nz);
        } 

        if (h_X != NULL)
            free(h_X);
        if (h_y != NULL)
            free(h_y);
    }

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
