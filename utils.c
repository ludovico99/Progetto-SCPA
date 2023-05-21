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

void free_X(int N, double ** X)
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

void free_y(int M, double ** y)
{

    for (int i = 0; i < M; i++)
    {
        if (y[i] != NULL)
            free(y[i]);
    }

    if (y != NULL)
        free(y);
}