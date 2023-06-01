
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include "include/header.h"

/*------------------------------------------------------- CSR ---------------------------------------------------------------------*/

/**
 * serial_product_CSR - Function that implements the sparse matrix - dense vector product with A in the CSR format
 * @param M: Number of rows of the matrix A
 * @param N: Number of columns of the matrix A, Number of rows of the matrix X
 * @param K: Number of columns of the matrix X
 * @param nz: Number of nz
 * @param as: Coefficient vector
 * @param ja: Column index vector
 * @param irp: Vector of the start index of each row
 * @param X: Matrix X
 * @param time: Pointer of a double representing the time
 *
 * Returns resulting matrix y of dimension (M * K)
 */

double **serial_product_CSR(int M, int N, int K, int nz, double *as, int *ja, int *irp, double **X, double *time)
{

    double **y = NULL;
    struct timespec start, stop;
    double partial_sum = 0.0;

    AUDIT printf("Computing serial product ...\n");

    /**
     * Allocating memory for the resulting matrix y
     */

    memory_allocation(double *, M, y);

    for (int i = 0; i < M; i++)
    {
        all_zeroes_memory_allocation(double, K, y[i]);
    }

    AUDIT printf("y correctly allocated ... \n");

    /**
     * Getting the elapsed time since—as described by POSIX—"some unspecified point in the past"
     */

    get_time(&start);

    /**
     * Starting serial sparse matrix - dense vector product
     */
    for (int i = 0; i < M; i++)
    {
        int start = irp[i]; // Starting index for the i-th row
        int end = 0;

        if (i < M - 1)
            end = irp[i + 1]; // Ending index for the i-th row
        else
            end = nz;

        for (int z = 0; z < K; z++)
        {

            double partial_sum = 0.0;

            for (int j = start; j < end; j++)
            {   
                if (as != NULL) // A is not a pattern matrix
                    partial_sum += as[j] * X[ja[j]][z];
                else
                    partial_sum += 1.0 * X[ja[j]][z];
            }
            y[i][z] = partial_sum;
        }
    }
    /**
     * Getting the elapsed time ssince—as described by POSIX—"some unspecified point in the past"
     */

    get_time(&stop);

    AUDIT printf("Completed serial product ...\n");

    /**
     * Computing elapsed time and GLOPS for the serial product
     */
    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL)
        *time = accum;

    AUDIT printf("ELAPSED TIME FOR SERIAL PRODUCT: %lf\n", accum);
    AUDIT printf("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    //print_y(M, K, y);
    return y;
}

/*------------------------------------------------------- ELLPACK ---------------------------------------------------------------------*/

/**
 * serial_product_ellpack - Function that implements the sparse matrix - dense vector product with A in the ELLPACK format with 0x0 padding
 * @param M: Number of rows of the matrix A
 * @param N: Number of columns of the matrix A, Number of rows of the matrix X
 * @param K: Number of columns of the matrix X
 * @param nz: Number of nz
 * @param max_nz_per_row: Maximum number of non-zeros between all rows
 * @param as: 2D array of coefficients
 * @param ja: 2D array of column indexes
 * @param X: Matrix X
 * @param time: Pointer of a double representing the time
 *
 * Returns resulting matrix y of dimension (M * K)
 */

double **serial_product_ellpack(int M, int N, int K, int nz, int max_nz_per_row, double **as, int **ja, double **X, double *time)
{

    double **y = NULL;
    struct timespec start, stop;

    AUDIT printf("Computing serial product ...\n");

    /**
     * Allocating memory for the resulting matrix y
     */

    memory_allocation(double *, M, y);

    for (int i = 0; i < M; i++)
    {
        all_zeroes_memory_allocation(double, K, y[i]);
    }
    AUDIT printf("y correctly allocated ... \n");

    /**
     * Getting the elapsed time since—as described by POSIX—"some unspecified point in the past"
     */
    get_time(&start);

    /**
     * Starting serial sparse matrix - dense vector product
     */

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
                if (as != NULL) // A is not a pattern matrix
                    y[i][z] += as[i][j] * X[ja[i][j]][z];
                else
                    y[i][z] += 1.0 * X[ja[i][j]][z];

                if (j < max_nz_per_row - 2)
                {
                    if (ja[i][j] == ja[i][j + 1]) // It means that the i-th row has no more zeros
                        break;
                }
            }
        }
    }

    /**
     * Getting the elapsed time ssince—as described by POSIX—"some unspecified point in the past"
     */

    get_time(&stop);

    AUDIT printf("Completed serial product ...\n");

    /**
     * Computing elapsed time and GLOPS for the serial product
     */

    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL)
        *time = accum;

    AUDIT printf("ELAPSED TIME FOR SERIAL PRODUCT: %lf\n", accum);
    AUDIT printf("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}

/**
 * serial_product_ellpack_no_zero_padding - Function that implements the sparse matrix - dense vector product with A in the ELLPACK format without 0x0 padding
 * @param M: Number of rows of the matrix A
 * @param N: Number of columns of the matrix A, Number of rows of the matrix X
 * @param K: Number of columns of the matrix X
 * @param nz: Number of nz
 * @param max_nz_per_row: Maximum number of non-zeros between all rows
 * @param as: 2D array of coefficients
 * @param ja: 2D array of column indexes
 * @param X: Matrix X
 * @param time: Pointer of a double representing the time
 *
 * Returns resulting matrix y of dimension (M * K)
 */

double **serial_product_ellpack_no_zero_padding(int M, int N, int K, int nz, int *nz_per_row, double **as, int **ja, double **X, double *time)
{

    double **y = NULL;
    struct timespec start, stop;

    AUDIT printf("Computing serial product ...\n");

    /**
     * Allocating memory for the resulting matrix y
     */

    memory_allocation(double *, M, y);

    for (int i = 0; i < M; i++)
    {
        all_zeroes_memory_allocation(double, K, y[i]);
    }

    AUDIT printf("y correctly allocated ... \n");
    /**
     * Getting the elapsed time ssince—as described by POSIX—"some unspecified point in the past"
     */
    get_time(&start);

    /**
     * Starting serial sparse matrix - dense vector product
     */

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
                if (as != NULL) // A is not a pattern matrix
                    y[i][z] += as[i][j] * X[ja[i][j]][z];
                else
                    y[i][z] += 1.0 * X[ja[i][j]][z];
            }
        }
    }
    AUDIT printf("Completed serial product ...\n");

    /**
     * Getting the elapsed time ssince—as described by POSIX—"some unspecified point in the past"
     */

    get_time(&stop);

    /**
     * Computing elapsed time and GLOPS for the serial product
     */

    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL)
        *time = accum;

    AUDIT printf("ELAPSED TIME FOR SERIAL PRODUCT: %lf\n", accum);
    AUDIT printf("GLOPS are %lf\n", compute_GFLOPS(K, nz, accum * 1e9));

    return y;
}