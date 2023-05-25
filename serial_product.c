
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include "header.h"

double **serial_product_CSR(int M, int N, int K, int nz, double *as, int *ja, int *irp, double **X, double *time)
{

    double **y = NULL;
    struct timespec start, stop;
    double partial_sum = 0.0;

    AUDIT printf("Computing serial product ...\n");

    memory_allocation(double *, M, y);


    for (int i = 0; i < M; i++)
    {   
        all_zeroes_memory_allocation(double, K, y[i]);

    }

    AUDIT printf("y correctly allocated ... \n");

    // calcola il prodotto matrice - multi-vettore

    if (clock_gettime(CLOCK_MONOTONIC, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < M; i++)
    {

        for (int z = 0; z < K; z++)
        {
            int start = irp[i];
            int end = irp[i + 1];

            if (i < M - 1 && start == end)
            {
                AUDIT printf("Row %d is the vector zero\n", i);
                y[i][z] = 0.0;
            }
            else
            {
                double partial_sum = 0.0;
                

                for (int j = start; (i < (M - 1) && j < end) || (i >= M - 1 && j < nz); j++)
                {
                    if (as != NULL)
                        partial_sum += as[j] * X[ja[j]][z];
                    else
                        partial_sum += 1.0 * X[ja[j]][z];
                }
                y[i][z] = partial_sum;
            }
        }
    }

    if (clock_gettime(CLOCK_MONOTONIC, &stop) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }
    AUDIT printf("Completed serial product ...\n");

    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL)
        *time = accum;

    AUDIT printf("ELAPSED TIME FOR SERIAL PRODUCT: %lf\n", accum);
    AUDIT printf("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}

double **serial_product_ellpack(int M, int N, int K, int nz, int max_nz_per_row, double **as, int **ja, double **X, double *time)
{

    double **y = NULL;
    struct timespec start, stop;

    AUDIT printf("Computing serial product ...\n");

    memory_allocation(double *, M, y);


    for (int i = 0; i < M; i++)
    {   
        all_zeroes_memory_allocation(double, K, y[i]);

    }
    AUDIT printf("y correctly allocated ... \n");
    // calcola il prodotto matrice - multi-vettore
    if (clock_gettime(CLOCK_MONOTONIC, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < M; i++)
    {

        for (int z = 0; z < K; z++)
        {

            for (int j = 0; j < max_nz_per_row; j++)
            {
                if (ja[i][j] == -1)
                {
                    y[i][z] = 0.0;
                    break;
                }
                if (as != NULL)
                    y[i][z] += as[i][j] * X[ja[i][j]][z];
                else
                    y[i][z] += 1.0 * X[ja[i][j]][z];

                if (j < max_nz_per_row - 2)
                {
                    if (ja[i][j] == ja[i][j + 1])
                        break;
                }
            }
        }
    }

    if (clock_gettime(CLOCK_MONOTONIC, &stop) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }
    AUDIT printf("Completed serial product ...\n");

    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL)
        *time = accum;

    AUDIT printf("ELAPSED TIME FOR SERIAL PRODUCT: %lf\n", accum);
    AUDIT printf("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}

double **serial_product_ellpack_no_zero_padding(int M, int N, int K, int nz, int *nz_per_row, double **as, int **ja, double **X, double *time)
{

    double **y = NULL;
    struct timespec start, stop;

    AUDIT printf("Computing serial product ...\n");

    memory_allocation(double *, M, y);

    for (int i = 0; i < M; i++)
    {   
        all_zeroes_memory_allocation(double, K, y[i]);

    }

    AUDIT printf("y correctly allocated ... \n");
    // calcola il prodotto matrice - multi-vettore
    if (clock_gettime(CLOCK_MONOTONIC, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < M; i++)
    {

        for (int z = 0; z < K; z++)
        {
            if (nz_per_row[i] == 0)
            {
                y[i][z] = 0.0;
                continue;
            }
            for (int j = 0; j < nz_per_row[i]; j++)
            {
                if (as != NULL)
                    y[i][z] += as[i][j] * X[ja[i][j]][z];
                else
                    y[i][z] += 1.0 * X[ja[i][j]][z];
            }
        }
    }
    AUDIT printf("Completed serial product ...\n");

    if (clock_gettime(CLOCK_MONOTONIC, &stop) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL)
        *time = accum;

    AUDIT printf("ELAPSED TIME FOR SERIAL PRODUCT: %lf\n", accum);
    AUDIT printf("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}