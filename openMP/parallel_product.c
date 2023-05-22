#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <stdio.h>

#include "../header.h"

double **parallel_product_CSR(int M, int N, int K, int nz, double *as_A, int *ja_A, int *irp_A, double **X,double * time, int nthread)
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

    // calcola il prodotto matrice - multi-vettore
    if (clock_gettime(CLOCK_MONOTONIC, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    if (M % nthread == 0)
    {
        chunk_size = M / nthread;
    }
    else
        chunk_size = M / nthread + 1;


#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthread) shared(y, as_A, X, ja_A, irp_A, M, K, nz, nthread, chunk_size) default(none)
    for (int i = 0; i < M; i++)
    {
        // #pragma omp parallel for schedule(static, K/8) num_threads(nthread) shared(y, as_A, X, ja_A, irp_A, M, K, nz, i) default(none)
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
                //printf("Computing y[%d][%d]\n", i, z);

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
    AUDIT printf("Completed parallel product ...\n");
    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL) *time = accum;

    AUDIT printf("ELAPSED TIME FOR PARALLEL PRODUCT: %lf\n", accum);
    AUDIT printf ("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}

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
    if (clock_gettime(CLOCK_MONOTONIC, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

     if (M % nthread == 0)
    {
        chunk_size = M / nthread;
    }
    else
        chunk_size = M / nthread + 1;

#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthread) shared(y, as, X, ja, M, K, max_nz_per_row, nthread, chunk_size) default(none)
    for (int i = 0; i < M; i++)
    {

        for (int z = 0; z < K; z++)
        {
            //printf("Computing y[%d][%d]\n", i, z);

            for (int j = 0; j < max_nz_per_row; j++)
            {      
                if (ja[i][j] == -1){
                    y[i][z] = 0.0;
                    break;
                }
                if (as != NULL)
                    y[i][z] += as[i][j] * X[ja[i][j]][z];
                else
                    y[i][z] += 1.0 * X[ja[i][j]][z];

                if (j < max_nz_per_row - 2)
                {
                    if (ja[i][j] == ja[i][j + 1]) break;
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
    if (time != NULL) *time = accum;

    AUDIT printf("ELAPSED TIME FOR PARALLEL PRODUCT: %lf\n", accum);
    AUDIT printf ("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}

double **parallel_product_ellpack_no_zero_padding(int M, int N, int K, int nz, int* nz_per_row, double **as, int **ja, double **X, double * time, int nthread)
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

     if (M % nthread == 0)
    {
        chunk_size = M / nthread;
    }
    else
        chunk_size = M / nthread + 1;

#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthread) shared(y, as, X, ja, M, K, nz_per_row, nthread, chunk_size) default(none)
    for (int i = 0; i < M; i++)
    {

        for (int z = 0; z < K; z++)
        {
            //AUDIT printf("Computing y[%d][%d]\n", i, z);
            if (nz_per_row[i] == 0) {
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
    //AUDIT printf("Completed parallel product ...\n");

    if (clock_gettime(CLOCK_MONOTONIC, &stop) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL) *time = accum;

    AUDIT printf("ELAPSED TIME FOR PARALLEL PRODUCT: %lf\n", accum);
    AUDIT printf ("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}