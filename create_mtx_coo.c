#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>

#include "lib/mmio.h"
#include "include/header.h"

/**
 * init_stream - Auxiliary function: Reopening the file and moving to the correct location
 *
 * Returns the FILE* of the file that has been opened
 */

static FILE *init_stream(void)
{
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    if ((f = fopen(filename, "r")) == NULL)
        exit(1);

    if (mm_read_banner(f, &matcode) != 0) // Read the banner to understand the type of the matrix
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) // Acquiring M, N ,nz for the matrix represented in the .mtx file
        exit(1);

    return f;
}

/**
 * coo_general - Function implemented to correctly read a matrix of type general
 * @param mode: It's a code that represents a matrix type in our application
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 */
void coo_general(int mode, int *M, int *N, int *nz, int **I, int **J, double **val)
{
    int ret_code;
    int chunk_size;
    double value;

    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0) // Acquiring M, N ,nz for the matrix represented in the .mtx file
    {
        printf("Error reading size file...\n");
        exit(1);
    }

    if (mode == GENERAL_PATTERN)
        printf("GENERAL-PATTERN MATRIX\n");
    else
        printf("GENERAL-REAL MATRIX\n");

    printf("total not zero: %d\n", *nz);

    /**
     * Allocating memory for the COO matrix representation
     */

    all_zeroes_memory_allocation(int, *nz, *I);
    all_zeroes_memory_allocation(int, *nz, *J);

    if (mode == GENERAL_PATTERN)
        *val = NULL;
    else
    {
        memory_allocation(double, *nz, *val);
    }
    /**
     * If the array is pattern each element has value 1.0 and is not stored in the .mtx file
     */
    if (mode == GENERAL_PATTERN)
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d\n", &((*I)[i]), &((*J)[i]));
            if (ret_code != 2)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            (*I)[i]--; /* adjust from 1-based to 0-based */
            (*J)[i]--;
        }
    }
    /**
     * If the array is not pattern each element has its own value and is stored in the .mtx file
     */
    else
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d %lg\n", &((*I)[i]), &((*J)[i]), &((*val)[i]));
            if (ret_code != 3)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            (*I)[i]--; /* adjust from 1-based to 0-based */
            (*J)[i]--;
        }
    }

    printf("COO Matrix Conversion completed\n");
}

/**
 * coo_symm - Function implemented to correctly read a matrix of type symmetric
 * @param mode: It's a code that represents a matrix type in our application
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 */

void coo_symm(int mode, int *M, int *N, int *nz, int **I, int **J, double **val)
{
    int ret_code;
    int real_nz;
    int row;    // Row index for the current nz
    int column; // Column index for the current nz
    int chunk_size;
    double value;

    /*
     * Reading the information about the dimensions of a sparse matrix in the COO format in a Matrix Market file
     */
    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0) // Acquiring M, N ,nz for the matrix represented in the .mtx file
    {
        printf("Reading Matrix Dimensions error\n");
        exit(1);
    }

    if (mode == SYMM_PATTERN)
        printf("SYMMETRIC-PATTERN MATRIX\n");

    else
        printf("SYMMETRIC-REAL MATRIX\n");

    printf("Initial nz for a symmetric matrix: %d\n", *nz);

    real_nz = *nz * 2;

    /**
     *  I subtract from double the number of non-zero (lower triangle diagonal), that have been read from the file,
     * the number of non-zero that are present on the diagonal.
     */

    if (mode == SYMM_PATTERN) /* If the array is pattern each element has value 1.0 and is not stored in the .mtx file */
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d\n", &row, &column); //Reading the .mtx file
            if (ret_code != 2)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            if (row == column)
            {
                real_nz--;
            }
        }
    }
    else /* If the array is not pattern each element has its own value nd is stored in the .mtx file */
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d %lg\n", &row, &column, &value); //Reading the .mtx file
            if (ret_code != 3)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            if (row == column)
            {
                real_nz--;
            }
        }
    }

    if (f != NULL)
        fclose(f);
    /**
     * Re-inizializing the stream since the file .mtx has been completely read
     */
    f = init_stream();

    /**
     * Allocating memory for the COO matrix representation
     */

    all_zeroes_memory_allocation(int, real_nz, *I);
    all_zeroes_memory_allocation(int, real_nz, *J);

    if (mode == SYMM_PATTERN)
        *val = NULL;
    else
    {
        memory_allocation(double, real_nz, *val);
    }

    int counter = 0;

    if (mode == SYMM_PATTERN)
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d\n", &((*I)[i]), &((*J)[i])); //Reading the .mtx file
            if (ret_code != 2)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            (*I)[i]--; /* adjust from 1-based to 0-based */
            (*J)[i]--;

            if ((*I)[i] != (*J)[i])
            {

                (*I)[*nz + counter] = (*J)[i];
                (*J)[*nz + counter] = (*I)[i];
                counter++;
            }
        }
    }
    else
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d %lg\n", &((*I)[i]), &((*J)[i]), &((*val)[i])); //Reading the .mtx file
            if (ret_code != 3)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            (*I)[i]--; /* adjust from 1-based to 0-based */
            (*J)[i]--;

            if ((*I)[i] != (*J)[i])
            {

                (*I)[*nz + counter] = (*J)[i];
                (*J)[*nz + counter] = (*I)[i];
                (*val)[*nz + counter] = (*val)[i];
                counter++;
            }
        }
    }
    /**
     * Consistency check: counter should be equal to real_nz - *nz, that is the number of non-zeros in the lower triangle
    */
    if (counter != real_nz - *nz)
    {
        printf("Number of elements that are not on the diagonal is wrong\n");
        exit(1);
    }

    *nz = real_nz;
    printf("TOTAL NZ for a symmetric matrix: %d\n", *nz);
}