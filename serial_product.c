
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include "header.h"

double **serial_product_CSR(int M, int N, int K, int nz, double *as_A, int *ja_A, int *irp_A, double **X, double *time)
{

    double **y = NULL;
    struct timespec start, stop;

    AUDIT printf("Computing serial product ...\n");
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

    for (int i = 0; i < M; i++)
    {

        for (int z = 0; z < K; z++)
        {   
            if (i == 0 && irp_A[i] == -1){
                AUDIT printf("Row 0 is the vector zero\n");
                y[i][z] = 0.0;
            }
            if (i > 0 && irp_A[i] == irp_A[i - 1])
            {
                AUDIT printf("Row %d is the vector zero\n", i);
                y[i][z] = 0.0;
            }
            else
            {
                // AUDIT printf("Computing y[%d][%d]\n", i, z);

                // if (i < (M - 1))
                //     AUDIT printf("Riga %d, id della colonna del primo nz della riga %d e id della colonna del primo nz zero della riga successiva %d\n", i, ja_A[irp_A[i]], ja_A[irp_A[i + 1]]);
                // else
                //     AUDIT printf("Riga %d, id della colonna del primo nz della riga %d\n", i, ja_A[irp_A[i]]);

                for (int j = irp_A[i]; (i < (M - 1) && j < irp_A[i + 1]) || (i >= M - 1 && j < nz); j++)
                {
                    if (as_A != NULL)
                        y[i][z] += as_A[j] * X[ja_A[j]][z];
                    else
                        y[i][z] += 1.0 * X[ja_A[j]][z];
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

    // for (int i = 0; i < M; i++)
    // {
    //     AUDIT printf("\n");
    //     for (int z = 0; z < K; z++)
    //     {
    //         AUDIT printf("y[%d][%d] = %lf ", i, z, y[i][z]);
    //     }
    // }

    // AUDIT printf("\n");
    return y;
}

double **serial_product_ellpack(int M, int N, int K, int max_nz_per_row, double **as, int **ja, double **X, double *time)
{

    double **y = NULL;
    struct timespec start, stop;

    AUDIT printf("Computing serial product ...\n");
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

    for (int i = 0; i < M; i++)
    {

        for (int z = 0; z < K; z++)
        {
            // AUDIT printf("Computing y[%d][%d]\n", i, z);

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

    // for (int i = 0; i < M; i++)
    // {
    //     printf("\n");
    //     for (int z = 0; z < K; z++)
    //     {
    //         printf("y[%d][%d] = %.20lf\t", i, z, y[i][z]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    return y;
}

double **serial_product(int M, int N, int K, double **A, double **X)
{
    double **y = NULL;

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

    // calcola il prodotto matrice - multi-vettore
    for (int i = 0; i < M; i++)
    {
        for (int z = 0; z < K; z++)
        {
            for (int j = 0; j < N; j++)
            {
                y[i][z] += A[i][j] * X[j][z];
            }
        }
    }
    return y;
}

double **serial_product_ellpack_no_zero_padding(int M, int N, int K, int *nz_per_row, double **as, int **ja, double **X, double *time)
{

    double **y = NULL;
    struct timespec start, stop;

    AUDIT printf("Computing serial product ...\n");
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

    for (int i = 0; i < M; i++)
    {

        for (int z = 0; z < K; z++)
        {
            // AUDIT printf("Computing y[%d][%d]\n", i, z);
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

    // for (int i = 0; i < M; i++)
    // {
    //     printf("\n");
    //     for (int z = 0; z < K; z++)
    //     {
    //         printf("y[%d][%d] = %.20lf\t", i, z, y[i][z]);
    //     }
    //     printf("\n");
    // }

    AUDIT printf("\n");
    return y;
}