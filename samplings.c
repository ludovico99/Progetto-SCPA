#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "header.h"

static double calculate_mean(double x, double mean, int n)
{
    mean += (x - mean) / n;
    return mean;
}

static double calculate_variance(double x, double mean, double variance, int n)
{
    if (n == 1)
    {
        return 0.0;
    }
    double delta = x - mean;
    mean = calculate_mean(x, mean, n);
    variance += delta * (x - mean);
    return variance / (n - 1);
}

#ifdef CSR
void computing_samplings_openMP(int M, int N, int *K, int nz, double *as, int *ja, int *irp, int max_nthread)
#elif ELLPACK 
void computing_samplings_openMP(int M, int N, int *K, int nz,  int *nz_per_row, double ** values, int **col_indices, int max_nthread)
#endif
{

    FILE *f_samplings;
    char *fn;

    double **X = NULL;
    double **y = NULL;
    double time = 0.0;
    double mean = 0.0;
    double variance = 0.0;
    double Gflops = 0.0;

// SAMPLING FOR THE SERIAL PRODUCT
#ifdef SAMPLING_SERIAL
#ifdef ELLPACK

    fn = "samplings_serial_ELLPACK.csv";
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "K,mean,variance\n");

#elif CSR

    fn = "samplings_serial_CSR.csv";
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "K,mean,variance\n");

#endif

#endif
// SAMPLING FOR THE PARALLEL PRODUCT
#ifdef SAMPLING_PARALLEL
#ifdef ELLPACK
    fn = "samplings_parallel_ELLPACK.csv";
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "K,num_threads,mean,variance\n");
#elif CSR
    fn = "samplings_parallel_CSR.csv";
    f_samplings = fopen(fn, "w+");
    fprintf(f_samplings, "K,num_threads,mean,variance\n");
#endif

#endif

    for (int k = 0; k < 7; k++)
    {
        create_dense_matrix(N, K[k], &X);
#ifdef SAMPLING_PARALLEL
        for (int num_thread = 1; num_thread <= max_nthread; num_thread++)
        {
#endif
            mean = 0.0;
            variance = 0.0;
            Gflops = 0.0;

            for (int curr_samp = 0; curr_samp < SAMPLING_SIZE; curr_samp++)
            {
#ifdef ELLPACK
#ifdef SAMPLING_PARALLEL
                y = parallel_product_ellpack_no_zero_padding(M, N, K[k], nz_per_row, values, col_indices, X, &time, num_thread);

#elif SAMPLING_SERIAL
                y = serial_product_ellpack_no_zero_padding(M, N, K[k], nz_per_row, values, col_indices, X, &time);
#endif

#elif CSR
#ifdef SAMPLING_PARALLEL
            y = parallel_product_CSR(M, N, K[k], nz, as, ja, irp, X, &time, num_thread);
#else
            y = serial_product_CSR(M, N, K[k], nz, as, ja, irp, X, &time);
#endif
#endif

                mean = calculate_mean(time, mean, curr_samp + 1);
                variance = calculate_variance(time, mean, variance, curr_samp + 1);
                Gflops = calculate_mean(compute_GFLOPS(K[k], nz, time * 1e9), Gflops, curr_samp + 1);

                free_y (M, y);
            }

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
        free_X(M, X);
    }


    if (f_samplings != stdin)
        fclose(f_samplings);
}