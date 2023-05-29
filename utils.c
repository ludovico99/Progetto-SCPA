#include "include/header.h"
#include "lib/mmio.h"
#include "stdlib.h"


void create_dense_matrix(int N, int K, double ***X)
{

    AUDIT printf("Creating dense matrix ...\n");
    memory_allocation(double *, N, *X);

    for (int j = 0; j < N; j++)
    {
        memory_allocation(double, K, (*X)[j]);

        for (int z = 0; z < K; z++)
        {
            (*X)[j][z] = 1.0;
        }
    }

    AUDIT printf("Completed dense matrix creation...\n");
}

/* Computazione della dimensione del chunk per parallelizzare */
int compute_chunk_size(int value, int nthread)
{
    int chunk_size;

    if (value % nthread == 0)
        chunk_size = value / nthread;
    else
        chunk_size = value / nthread + 1;
    /*
     * 16 è pari al numero di interi in una linea di cache.
     * E' anche multiplo di 8, che è il numero di double in una cache line.
     */
    chunk_size = chunk_size - chunk_size % (16); // E' il numero intero piu vicino a chunk size multiplo di 16.
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

#ifdef CUDA
double *convert_2D_to_1D(int M, int K, double **A)
{   
    double * ret = NULL;
    all_zeroes_memory_allocation(double, M*K, ret);

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

    unsigned long sum_nz = 0;

    double * ret = NULL;
    all_zeroes_memory_allocation(double ,nz, ret);

    for (int i = 0; i < M; i++)
    {
        if (nz_per_row[i] == 0)
            continue;
        for (int j = 0; j < nz_per_row[i]; j++)
        {
            if (A[i] != NULL)
                ret[sum_nz + j] = A[i][j];
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

    int * ret = NULL;
    all_zeroes_memory_allocation(int ,nz, ret);

    for (int i = 0; i < M; i++)
    {
        if (nz_per_row[i] == 0)
            continue;
        for (int j = 0; j < nz_per_row[i]; j++)
        {
            if (A[i] != NULL)
                ret[sum_nz + j] = A[i][j];
        }
        sum_nz += nz_per_row[i];

        if (A[i] != NULL)
            free(A[i]);
    }
    if (A != NULL)
        free(A);
    return ret;
}
#endif

int *compute_sum_nz(int M, int *nz_per_row)
{

    printf("Computing sum_nz\n");

    int * ret = NULL;
    all_zeroes_memory_allocation(int ,M, ret);

    ret[1] = nz_per_row[0];
    for (int i = 2; i < M; i++)
    {
        ret[i] = ret[i - 1] + nz_per_row[i - 1];
    }
    return ret;
}

void print_y(int M, int K, double **y)
{
    for (int i = 0; i < M; i++)
    {
        printf("\n");
        for (int z = 0; z < K; z++)
        {
            printf("y[%d][%d] = %.20lf\t", i, z, y[i][z]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_y_GPU(int M, int K, double *y)
{

    for (int i = 0; i < M; i++)
    {
        printf("\n");
        for (int z = 0; z < K; z++)
        {
            printf("y[%d][%d] = %.70lf\t", i, z, y[i * K + z]);
        }
        printf("\n");
    }
    printf("\n");
}

double* transpose(int N, int K, double **A)
{   
    double *ret = NULL;

    printf ("Computing transpose for X ...\n");
 
    all_zeroes_memory_allocation(double ,N*K, ret);
    
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            ret[j * N + i] = A[i][j];
        }
    }

    return ret;
}

void get_time (struct timespec *time){

   if (clock_gettime(CLOCK_MONOTONIC, time) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

}