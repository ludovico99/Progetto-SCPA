#include "header.h"
#include "mmio.h"
#include "stdlib.h"
/* Computazione della dimensione del chunk per parallelizzare */
int compute_chunk_size(int value, int nthread)
{
    int chunk_size;

    if (value % nthread == 0)
        chunk_size = value / nthread;
    else
        chunk_size = value / nthread + 1;

    return chunk_size;
}

double compute_GFLOPS(int k, int nz, double time)
{

    double num = 2 * nz;
    return (double)((num / time) * k);
}

void free_CSR_data_structures(double *as, int *ja, int *irp)
{

    printf("Freeing CSR data structures...\n");
    if (as != NULL)
        free(as);
    if (ja != NULL)
        free(ja);
    if (irp != NULL)
        free(irp);
}

void free_ELLPACK_data_structures(int M, double **values, int **col_indices)
{

    printf("Freeing ELLPACK data structures...\n");

    for (int i = 0; i < M; i++)
    {
        if (values[i] != NULL)
            free(values[i]);
        if (col_indices[i] != NULL)
            free(col_indices[i]);
    }

    if (values != NULL)
        free(values);
    if (col_indices != NULL)
        free(col_indices);
}

void free_X(int N, double **X)
{

    printf("Freeing matrix X...\n");

    for (int i = 0; i < N; i++)
    {
        if (X[i] != NULL)
            free(X[i]);
    }

    if (X != NULL)
        free(X);
}

void free_y(int M, double **y)
{

    for (int i = 0; i < M; i++)
    {
        if (y[i] != NULL)
            free(y[i]);
    }

    if (y != NULL)
        free(y);
}

double *convert_2D_to_1D(int M, int K, double **A)
{

    double *ret = (double *)calloc(M * K, sizeof(double));
    if (ret == NULL)
    {
        printf("Malloc failed for ret ...");
        exit(1);
    }
    printf("Starting 2D conversion in 1D\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            if (A[i] != NULL)
                ret[i * K + j] = A[i][j];
        }
        if (A[i] != NULL)
            free(A[i]);
    }
    if (A != NULL)
        free(A);
    return ret;
}

double *convert_2D_to_1D_per_ragged_matrix(int M, int nz, int *nz_per_row, double **A)
{

    printf("Starting 2D conversion in 1D for a ragged matrix\n");

    unsigned long  sum_nz = 0;

    double *ret = (double *)calloc(M * nz, sizeof(double));
    if (ret == NULL)
    {
        printf("Malloc failed for ret ...\n");
        exit(1);
    }

    for (int i = 0; i < M; i++)
    {
        if (nz_per_row[i] == 0)
            continue;
        for (int j = 0; j < nz_per_row[i]; j++)
        {
            if (A[i] != NULL)
                ret[i * sum_nz + j] = A[i][j];
        }
        sum_nz += nz_per_row[i];
        if (A[i] != NULL)
            free(A[i]);
    }
    if (A != NULL)
        free(A);
    return ret;
}

int *convert_2D_to_1D_per_ragged_matrix(int M, int nz, int *nz_per_row, int **A)
{
    unsigned long sum_nz = 0;
    printf("Starting 2D conversion in 1D for a ragged matrix\n");

    int *ret = (int *)calloc(M * nz, sizeof(int));
    if (ret == NULL)
    {
        printf("Malloc failed for ret ...");
        exit(1);
    }

    for (int i = 0; i < M; i++)
    {
        if (nz_per_row[i] == 0)
            continue;
        for (int j = 0; j < nz_per_row[i]; j++)
        {
            if (A[i] != NULL)
                ret[i * sum_nz + j] = A[i][j];
        }
        sum_nz += nz_per_row[i];

        if (A[i] != NULL)
            free(A[i]);
    }
    if (A != NULL)
        free(A);
    return ret;
}

int *compute_sum_nz(int M, int *nz_per_row)
{

    printf("Computing sum_nz\n");

    int *ret = (int *)calloc(M, sizeof(int));
    if (ret == NULL)
    {
        printf("Malloc failed for ret ...");
        exit(1);
    }
    ret[1] = nz_per_row[0];
    for (int i = 2; i < M; i++)
    {
        ret[i] = ret[i - 1] + nz_per_row[i - 1];
    }
    return ret;
}