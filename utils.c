#include "include/header.h"
#include "lib/mmio.h"
#include <stdlib.h>
/**
 * create_dense_matrix - Create a dense matrix (assuming the number of non-zeros is zero) of all 1.0
 * @param N: Number of rows
 * @param K: Number of columns
 * @param nz: Number of nz
 * @param X: Pointer to a 2D array of double
 */

void create_dense_matrix(int N, int K, double ***X)
{

    AUDIT printf("Creating dense matrix ...\n");
    memory_allocation(double *, N, *X);
    //srand (time(NULL));
    for (int j = 0; j < N; j++)
    {
        memory_allocation(double, K, (*X)[j]);

        for (int z = 0; z < K; z++)
        {
            (*X)[j][z] =  (double)(rand()%10 + 1);
        }
    }

    AUDIT printf("Completed dense matrix creation...\n");
}

/**
 * compute_chunk_size - Computation of the size of the chunk to be assigned to each thread
 * @param value: Iteration space dimension
 * @param nthread: number of processors available to the device.
 *
 * Returns the value chunk size computed
 */

int compute_chunk_size(int value, int nthread)
{
    int chunk_size;

    if (value % nthread == 0)
        chunk_size = value / nthread;
    else
        chunk_size = value / nthread + 1;
    /*
     * 16 is equal to the number of integers in a cache line.
     * It is also a multiple of 8, which is the number of doubles in a cache line.
     */
    if (chunk_size > 16) chunk_size = chunk_size - chunk_size % (16); // It is the integer closest to the chunk size multiple of 16.
    return chunk_size;
}

/**
 * compute_GFLOPS - Computing the GLOPS
 * @param k: Number of columns of the dense matrix
 * @param nz: Number of nz
 * @param time: time in nano-secons
 * Returns the computed GLOPS value
 */
double compute_GFLOPS(int k, int nz, double time)
{

    double num = 2 * nz;
    return (double)((num / time) * k);
}

/**
 * free_CSR_data_structures - Function that frees the data structures allocated for the CSR format
 * @param as: Coefficient vector
 * @param ja: column index vector
 * @param ja: vector of the start index of each row
 */

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

/**
 * free_ELLPACK_data_structures - Function that frees the data structures allocated for the ELLPACK format
 * @param M: Number of rows
 * @param values: 2D array of coefficients
 * @param col_indices: 2D array of column indexes
 */

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

/**
 * free_X - Function that frees the X matrix
 * @param N: Number of rows
 * @param X: Matrix X stored by rows
 */

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

/**
 * free_X - Function that frees the y matrix
 * @param M: Number of rows
 * @param X: Matrix y stored by rows
 */

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

/**
 * convert_2D_to_1D - Function that converts a 2D in a 1D matrix
 * @param M: Number of rows
 * @param K: Number of columns
 * @param A: Matrix to be converted
 */

double *convert_2D_to_1D(int M, int K, double **A)
{
    double *ret = NULL;
    all_zeroes_memory_allocation(double, M *K, ret);

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

/**
 * convert_2D_to_1D_per_ragged_matrix - Function that converts a 2D in a 1D irregular matrix
 * @param M: Number of rows
 * @param nz: Number of non-zeroes
 * @param nz_per_row: Array of number of non-zero per row
 * @param A: Matrix to be converted
 */

double *convert_2D_to_1D_per_ragged_matrix(int M, int nz, int *nz_per_row, double **A)
{

    printf("Starting 2D conversion in 1D for a ragged matrix\n");

    unsigned long sum_nz = 0;

    double *ret = NULL;
    all_zeroes_memory_allocation(double, nz, ret);

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

/**
 * convert_2D_to_1D_per_ragged_matrix - Function that converts a 2D in a 1D irregular matrix
 * @param M: Number of rows
 * @param nz: Number of non-zeroes
 * @param nz_per_row: Array of number of non-zero per row
 * @param A: Matrix to be converted
 */

int *convert_2D_to_1D_per_ragged_matrix(int M, int nz, int *nz_per_row, int **A)
{
    unsigned long sum_nz = 0;
    printf("Starting 2D conversion in 1D for a ragged matrix\n");

    int *ret = NULL;
    all_zeroes_memory_allocation(int, nz, ret);

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

/**
 * compute_sum_nz - Function that computes, for the i-th row, the sum of non-zeroes up to the i-th row
 * @param M: Number of rows
 * @param nz_per_row: Array of number of non-zero per row
 *
 * Returns an array that represents the number of non-zeroes up to the i-th row
 */

int *compute_sum_nz(int M, int *nz_per_row)
{

    printf("Computing sum_nz\n");

    int *ret = NULL;
    all_zeroes_memory_allocation(int, M, ret);

    ret[1] = nz_per_row[0];
    for (int i = 2; i < M; i++)
    {
        ret[i] = ret[i - 1] + nz_per_row[i - 1];
    }
    return ret;
}

/**
 * print_y - DEBUG FUNCTION: It prints the matrix y
 * @param M: Number of rows
 * @param K: Number of non-zeroes
 * @param y: Matrix to be printed
 *
 */

void print_y(int M, int K, double **y)
{
    for (int i = 0; i < M; i++)
    {
        printf("\n");
        for (int z = 0; z < K; z++)
        {
            printf("y[%d][%d] = %.70lf\t", i, z, y[i][z]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * print_y_GPU - DEBUG FUNCTION: It prints the matrix y for a matrix in the "GPU" (linear) format
 * @param M: Number of rows
 * @param K: Number of non-zeroes
 * @param y: Matrix to be printed
 *
 */

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

/**
 * transpose - Calculate the transpose of the input matrix
 * @param N: Number of rows
 * @param K: Number of non-zeroes
 * @param A: Matrix to be transposed
 *
 */

double *transpose(int N, int K, double **A)
{
    double *ret = NULL;

    printf("Computing transpose for X ...\n");

    all_zeroes_memory_allocation(double, N *K, ret);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            ret[j * N + i] = A[i][j];
        }
    }

    return ret;
}

/**
 * get_time - It computes the time elapsed since the start of the program
 * @param time: pointer to the struct timespec
 *
 */

void get_time(struct timespec *time)
{

    if (clock_gettime(CLOCK_MONOTONIC, time) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }
}