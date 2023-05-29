#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include "include/header.h"

/*------------------------------------------------------- CSR ---------------------------------------------------------------------*/

/**
 * coo_to_CSR_serial - Serial version for converting a sparse matrix from COO format to CSR format
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 * @param as: Pointer to coefficient vector
 * @param ja: Pointer to the column index vector
 * @param irp: Pointer to the vector of the start index of each row
 */

int* coo_to_CSR_serial(int M, int N, int nz, int *I, int *J, double *val, double **as, int **ja, int **irp)
{
    int *nz_per_row = NULL;
    int chunk_size = 0;

    printf("Starting serial CSR conversion ...\n");

    all_zeroes_memory_allocation(int, M, nz_per_row);

    /**
     * Allocating memory for the CSR vectors
     */

    if (val != NULL)
    {
        memory_allocation(double, nz, *as);
    }

    memory_allocation(int, nz, *ja);

    memory_allocation(int, M, *irp);
    memset(*irp, -1, sizeof(int) * M);

    printf("Counting number of non-zero entries in each row...\n");

    /**
     * Computing the number of not zero for each row
     */

    for (int i = 0; i < nz; i++)
    {
        nz_per_row[I[i]]++;
    }

    printf("Filling CSR data structures ... \n");

    /**
     * Filling the vector irp
     */
    (*irp)[0] = 0;

    for (int i = 0; i < M - 1; i++)
    {
        (*irp)[i + 1] = (*irp)[i] + nz_per_row[i];
    }

    int offset = 0;
    int row;
    int idx;

    for (int i = 0; i < nz; i++)
    {
        row = I[i];
        idx = (*irp)[row];

        (*ja)[idx] = J[i];
        if (val != NULL)
            (*as)[idx] = val[i];

        (*irp)[row]++;
    }

    /**
     * Reset row pointers
     */
    for (int i = M - 1; i > 0; i--)
    {
        (*irp)[i] = (*irp)[i - 1];
    }

    (*irp)[0] = 0;

    /**
     * Freeing memory
     */

    printf("Freeing COO data structures...\n");
    if (I != NULL)
        free(I);
    if (J != NULL)
        free(J);
    if (val != NULL)
        free(val);

    printf("Completed serial CSR conversion ...\n");
    return nz_per_row;
}

/*------------------------------------------------------- ELLPACK ---------------------------------------------------------------------*/

/**
 * coo_to_ellpack_serial - serial version with 0x0 padding for converting a sparse matrix from COO format to ELLPACK format
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 * @param values: Pointer to the 2D array of coefficients
 * @param col_indices: Pointer to the 2D array of column indexes
 */

int coo_to_ellpack_serial(int M, int N, int nz, int *I, int *J, double *val, double ***values, int ***col_indices)
{
    int i, j, k;
    int max_nz_per_row = 0;

    printf("ELLPACK serial started...\n");

    /**
     * Calculates the maximum number of non-zero elements across all rows
     */
    for (int i = 0; i < M; i++)
    {
        int nz_in_row = 0;
        for (int j = 0; j < nz; j++)
        {
            if (I[j] == i)
                nz_in_row++;
        }
        if (nz_in_row > max_nz_per_row)
            max_nz_per_row = nz_in_row;
    }

    printf("MAX_NZ_PER_ROW is %d\n", max_nz_per_row);

    /**
     * Allocating memory for the ELLPACK 2D data structures
     */
    if (val != NULL)
    {
        memory_allocation(double *, M, *values);

        for (int k = 0; k < M; k++)
        {
            memory_allocation(double, max_nz_per_row, (*values)[k]);
        }
    }

    memory_allocation(int *, M, *col_indices);

    for (int k = 0; k < M; k++)
    {
        memory_allocation(int, max_nz_per_row, (*col_indices)[k]);
    }

    printf("Malloc for ELLPACK data structures completed\n");

    /**
     * Fills ELLPACK arrays with values and corresponding column indexes
     */

    for (int i = 0; i < M; i++)
    {
        int offset = 0;
        for (int j = 0; j < nz; j++)
        {
            if (I[j] == i)
            {
                if (val != NULL)
                    (*values)[i][offset] = val[j];
                (*col_indices)[i][offset] = J[j];
                offset++; //Throught offset it iterates on the next element for each row to be setted
            }
        }

        for (k = offset; k < max_nz_per_row; k++)
        {
            if (val != NULL)
                (*values)[i][k] = 0.0;
            if (offset != 0)
                (*col_indices)[i][k] = (*col_indices)[i][offset - 1];
            else
                (*col_indices)[i][k] = -1; //The i-th row has 0 not zero elements
        }
    }

    /**
     * Freeing memory
     */

#ifndef CHECK_CONVERSION
    printf("Freeing COO data structures...\n");
    if (I != NULL)
        free(I);
    if (J != NULL)
        free(J);
    if (val != NULL)
        free(val);
#endif

    return max_nz_per_row;
}
