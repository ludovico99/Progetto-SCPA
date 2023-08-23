#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

#include "../include/header.h"

/**
 * NB: The signature of the following function changes according to CSR or ELLPACK format
*/
#ifdef CSR 
/**
 *
 * computing_samplings_openMP - Function that computs for a fixed number of samplings, the mean and the variance of the times as the number of threads and K vary.
 * The overall results are written in a proper .csv file.
 * 
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param as: Coefficient vector
 * @param ja: Column index vector
 * @param irp: Vector of the start index of each row
 * @param max_nthread: Number of processors available to the device.

 */

void computing_samplings_openMP(int M, int N, int nz, double *as, int *ja, int *irp, int max_nthread)
#elif ELLPACK
/**
 *
 * computing_samplings_openMP - Function that computs for a fixed number of samplings, the mean and the variance of the times as the number of threads and K vary.
 * The overall results are written in a proper .csv file.
 * 
 *
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param nz_per_row: Array containing the number of non-zeroes per row
 * @param values: 2D array of coefficients
 * @param col_indices: 2D array of column indexes
 * @param max_nthread: Number of processors available to the device.*/

void computing_samplings_openMP(int M, int N, int nz, int *nz_per_row, double **values, int **col_indices, int max_nthread)
#endif
{

    FILE *f_samplings;
    /**
     * Name of the file to be created and written to
    */
    char fn[100]; 

    double **X = NULL;
    double **y = NULL;

    double time = 0.0;
    double mean = 0.0;
    double M2 = 0.0;
    double variance = 0.0;
    double Gflops = 0.0;
    double curr_Gflops = 0.0;

    int K[] = {3, 4, 8, 12, 16, 32, 64};

    int curr_samp = 0;

    char *token;
    token = strtok(filename,"/");
    token = strtok(NULL, "/");

// SAMPLING FOR THE SERIAL PRODUCT
#ifdef SAMPLING_SERIAL
#ifdef ELLPACK

    sprintf (fn, "plots/samplings_ELLPACK_CPU_serial_%s.csv", token);
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "K,mean,variance\n");

#elif CSR
    sprintf (fn, "plots/samplings_CSR_CPU_serial_%s.csv", token);
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "K,mean,variance\n");

#endif

#endif
// SAMPLING FOR THE PARALLEL PRODUCT
#ifdef SAMPLING_PARALLEL
#ifdef ELLPACK

    sprintf (fn, "plots/samplings_ELLPACK_CPU_parallel_%s.csv", token);
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "K,num_threads,mean,variance\n");
#elif CSR
   
    sprintf (fn, "plots/samplings_CSR_CPU_parallel_%s.csv", token);
    printf("fn %s\n", fn);
    fflush(stdout);
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "K,num_threads,mean,variance\n");
#endif

#endif

    for (int k = 0; k < sizeof(K)/sizeof(int); k++)
    {   
        /**
         * Creating the X matrix with its number of columns specified by K[k]
        */
        create_dense_matrix(N, K[k], &X);
#ifdef SAMPLING_PARALLEL
        for (int num_thread = 1; num_thread <= max_nthread; num_thread++)
        {
#endif      
            /**
             * Resetting the average stats
            */
            mean = 0.0;
            M2 = 0.0;
            Gflops = 0.0;
            variance = 0.0;
            curr_Gflops = 0.0;

            for (curr_samp = 0; curr_samp < SAMPLING_SIZE; curr_samp++)
            {
#ifdef ELLPACK
#ifdef SAMPLING_PARALLEL
                y = parallel_product_ellpack_no_zero_padding(M, N, K[k], nz, nz_per_row, values, col_indices, X, &time, num_thread);

#elif SAMPLING_SERIAL
                y = serial_product_ellpack_no_zero_padding(M, N, K[k], nz, nz_per_row, values, col_indices, X, &time);
#endif

#elif CSR
#ifdef SAMPLING_PARALLEL
            y = parallel_product_CSR(M, N, K[k], nz, as, ja, irp, X, &time, num_thread);
#else
            y = serial_product_CSR(M, N, K[k], nz, as, ja, irp, X, &time);
#endif
#endif
                /**
                * Welford's one-pass algorithm is an efficient method for computing mean and variance in a single pass over a sequence of values.
                * It achieves this by updating the mean and variance incrementally as new values are encountered.
                */
            
                curr_Gflops = compute_GFLOPS(K[k], nz, time * 1e9);
                
                M2 = calculate_M2(time, mean, M2, curr_samp + 1);

                mean = calculate_mean(time, mean, curr_samp + 1);

                Gflops = calculate_mean(curr_Gflops, Gflops, curr_samp + 1);

                /**
                 * Freeing the resulting matrix y
                */
                free_y(M, y);
            }

            /*After processing all values, the variance can be calculated as M2 / (n).*/
            variance = M2 / SAMPLING_SIZE;
            /**
             * Writing in the file the overall mean and variance
            */
#ifdef SAMPLING_PARALLEL
            printf("MEAN in seconds for K %d, num_thread %d is %lf, GLOPS are %lf\n", K[k], num_thread, mean, Gflops);
            fprintf(f_samplings, "%d,%d,%lf,%.20lf\n", K[k], num_thread, mean, variance);
            fflush(f_samplings);
        }
#elif SAMPLING_SERIAL
        printf("MEAN in seconds for K %d is %lf, GLOPS are %lf\n", K[k], mean, Gflops);
        fprintf(f_samplings, "%d,%lf,%.20lf\n", K[k], mean, variance);
        fflush(f_samplings);
#endif  
        /**
         * Freeing the dense matrix X
        */
        free_X(N, X);
    }
    /**
     * Closing the file
    */
    if (f_samplings != stdin)
        fclose(f_samplings);
}