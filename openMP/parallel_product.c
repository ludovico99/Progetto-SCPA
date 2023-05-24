#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <stdio.h>

#include "../header.h"

double **parallel_product_CSR(int M, int N, int K, int nz, double *as, int *ja, int *irp, double **X, double *time, int nthread)
{

    double **y = NULL;
    int chunk_size;
    struct timespec start, stop;

    AUDIT printf("Computing parallel product ...\n");
    y = (double **)malloc(M * sizeof(double *));
    if (y == NULL)
    {
        printf("Errore malloc per y\n");
        exit(1);
    }

    for (int i = 0; i < M; i++)
    {
        y[i] = (double *)malloc(K * sizeof(double));
        if (y[i] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
        for (int z = 0; z < K; z++)
        {
            y[i][z] = 0.0;
        }
    }
    AUDIT printf("y correctly allocated ... \n");

    chunk_size = compute_chunk_size(M, nthread);

    // calcola il prodotto matrice - multi-vettore
    if (clock_gettime(CLOCK_MONOTONIC, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthread) shared(y, as, ja, irp, X, M, K, nz, nthread, chunk_size) default(none)
    for (int i = 0; i < M; i++)
    {
        for (int z = 0; z < K; z++)
        {
            int start = irp[i];
            int end = irp[i + 1];

            if (i == 0 && start == -1)
            {
                printf("Row 0 is the vector zero\n");
                y[i][z] = 0.0;
            }
            else if (i > 0 && start == irp[i - 1])
            {
                printf("Row %d is the vector zero\n", i);
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
    AUDIT printf("Completed parallel product ...\n");

    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL)
        *time = accum;

    AUDIT printf("ELAPSED TIME FOR PARALLEL PRODUCT: %lf\n", accum);
    AUDIT printf("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}
// Ellpack parallel product per ELLPACK con padding di zero
double **parallel_product_ellpack(int M, int N, int K, int nz, int max_nz_per_row, double **as, int **ja, double **X, double *time, int nthread)
{

    double **y = NULL;
    struct timespec start, stop;
    int chunk_size = 0;

    AUDIT printf("Computing parallel product ...\n");
    y = (double **)malloc(M * sizeof(double *));
    if (y == NULL)
    {
        printf("Errore malloc per y\n");
        exit(1);
    }

    for (int i = 0; i < M; i++)
    {
        y[i] = (double *)malloc(K * sizeof(double));
        if (y[i] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
        for (int z = 0; z < K; z++)
        {
            y[i][z] = 0.0;
        }
    }
    AUDIT printf("y correctly allocated ... \n");
    // calcola il prodotto matrice - multi-vettore

    chunk_size = compute_chunk_size(M, nthread);

    if (clock_gettime(CLOCK_MONOTONIC, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthread) shared(y, M, K, max_nz_per_row, as, X, ja, nthread, chunk_size) default(none)
    for (int i = 0; i < M; i++)
    {

        for (int z = 0; z < K; z++)
        {
            double partial_sum = 0.0;
            for (int j = 0; j < max_nz_per_row; j++)
            {
                if (ja[i][j] == -1)
                {
                    y[i][z] = 0.0;
                    break;
                }

                if (as != NULL)
                    partial_sum += as[i][j] * X[ja[i][j]][z];
                else
                    partial_sum += 1.0 * X[ja[i][j]][z];

                y[i][z] = partial_sum;

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
    AUDIT printf("Completed parallel product ...\n");

    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL)
        *time = accum;

    AUDIT printf("ELAPSED TIME FOR PARALLEL PRODUCT: %lf\n", accum);
    AUDIT printf("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}

double **parallel_product_ellpack_no_zero_padding(int M, int N, int K, int nz, int *nz_per_row, double **as, int **ja, double **X, double *time, int nthread)
{

    double **y = NULL;
    struct timespec start, stop;
    int chunk_size = 0;

    AUDIT printf("Computing parallel product ...\n");
    y = (double **)malloc(M * sizeof(double *));
    if (y == NULL)
    {
        printf("Errore malloc per y\n");
        exit(1);
    }

    for (int i = 0; i < M; i++)
    {
        y[i] = (double *)malloc(K * sizeof(double));
        if (y[i] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
        for (int z = 0; z < K; z++)
        {
            y[i][z] = 0.0;
        }
    }
    AUDIT printf("y correctly allocated ... \n");
    // calcola il prodotto matrice - multi-vettore
    if (clock_gettime(CLOCK_MONOTONIC, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    chunk_size = compute_chunk_size(M, nthread);

#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthread) shared(y, M, K, nz_per_row, as, X, ja, nthread, chunk_size) default(none)
    for (int i = 0; i < M; i++)
    {
        for (int z = 0; z < K; z++)
        {
            int end = nz_per_row[i];
            if (end == 0)
                y[i][z] = 0.0;

            else
            {
                double partial_sum = 0.0;
                for (int j = 0; j < end; j++)
                {
                    if (as != NULL)
                        partial_sum += as[i][j] * X[ja[i][j]][z];
                    else
                        partial_sum += 1.0 * X[ja[i][j]][z];
                }
                y[i][z] = partial_sum;
            }
        }
    }
    // AUDIT printf("Completed parallel product ...\n");

    if (clock_gettime(CLOCK_MONOTONIC, &stop) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL)
        *time = accum;

    AUDIT printf("ELAPSED TIME FOR PARALLEL PRODUCT: %lf\n", accum);
    AUDIT printf("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}