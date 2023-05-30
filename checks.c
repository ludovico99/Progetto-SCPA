#include <stdio.h>
#include <stdlib.h>
#include "include/header.h"
#include <math.h>

#define TOLLERANZA 2.22e-16

#ifdef ELLPACK

/**
 * compare_conversion_algorithms_ellpack - It compares two conversion algorithm implemented for ELLPACK
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 * 
 * Returns 0 if the 2 conversions produced are equal, otherwise returns zero
 */

int compare_conversion_algorithms_ellpack(int M, int N, int nz, int *I, int *J, double *val, int nthread)
{
    double **values_A = NULL;
    double **values_B = NULL;

    int **col_indices_A = NULL;
    int **col_indices_B = NULL;

    int *nz_per_row_A = NULL;
    int *nz_per_row_B = NULL;

    /**
     * Executing the two different conversions 
    */
    nz_per_row_A = coo_to_ellpack_no_zero_padding_parallel_optimization(M, N, nz, I, J, val, &values_A, &col_indices_A, nthread);
    nz_per_row_B = coo_to_ellpack_no_zero_padding_parallel(M, N, nz, I, J, val, &values_B, &col_indices_B, nthread);

    /**
     * Firstly, it compares if the two array of the number of not zero for each rows are equal
    */
    for (int i = 0; i < M; i++)
    {
        if (nz_per_row_A[i] != nz_per_row_B[i])
        {
            printf("Il numero di non zeri per la riga %d è diverso nelle due conversioni\n", i);
            return 1;
        }
    }
    /**
     * Comparing the arrays (values and col_indices) that represent the matrix in the ELLPACK format, returned by the previuos algorithms 
    */
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < nz_per_row_A[i]; j++)
        {
            int found = 0;
            for (int k = 0; k < nz_per_row_A[i]; k++)
            {
                if (col_indices_A[i][j] == col_indices_B[i][k]) 
                {
                    found++;
                    if (found > 1) //It means that there are at least two elements with the same column for a fixed row i
                    {
                        printf("The two conversions are different (found > 1)\n");
                        return 1;
                    }
                    if (values_A[i][j] != values_B[i][k])  //The two elements with same row and column index are different 
                    {
                        printf("The two conversions are different\n");
                        return 1;
                    }
                }
            }
            if (found == 0)
            {
                printf("The two conversions are different (found = 0)\n");
                return 1;
            }
        }
    }

    /**
     * Starting freeing memory
    */

    if (I != NULL)
        free(I);
    if (J != NULL)
        free(J);
    if (val != NULL)
        free(val);

    free_ELLPACK_data_structures(M, values_A, col_indices_A);
    free_ELLPACK_data_structures(M, values_B, col_indices_B);

    if (nz_per_row_A != NULL)
        free(nz_per_row_A);
    if (nz_per_row_B != NULL)
        free(nz_per_row_B);

    printf("Same ELLPACK conversions\n");

    return 0;
}
#endif

#ifdef CSR

/**
 * compare_conversion_algorithms_csr - It compares two conversion algorithm implemented for CSR
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 * 
 * Returns 0 if the 2 conversions produced are equal, otherwise returns zero
 */

int compare_conversion_algorithms_csr(int M, int N, int nz, int *I, int *J, double *val, int nthread)
{
    double *as_A = NULL;
    double *as_B = NULL;

    int *ja_A = NULL;
    int *ja_B = NULL;

    int *irp_A = NULL;
    int *irp_B = NULL;

    int *nz_per_row_A = NULL;
    int *nz_per_row_B = NULL;
    /**
     * Executing the 2 conversion algorithms we wants to compare
    */
    nz_per_row_A = coo_to_CSR_parallel_optimization(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A, nthread);
    nz_per_row_B = coo_to_CSR_parallel(M, N, nz, I, J, val, &as_B, &ja_B, &irp_B, nthread);


      /**
     * Firstly, it compares if the two array of the number of not zero for each rows are equal
    */
    for (int i = 0; i < M; i++)
    {
        if (nz_per_row_A[i] != nz_per_row_B[i])
        {
            printf("Errore nella conversione numero di righe differenti\n");
            exit(1);
        }
    }

     /**
     * Comparing the arrays that represent the matrix in the CSR format, returned by the previuos algorithms 
    */
    for (int i = 0; i < M; i++)
    {
        for (int j = irp_A[i]; (i < (M - 1) && j < irp_A[i + 1]) || (i >= M - 1 && j < nz); j++)
        {
            int found = 0;
            for (int k = irp_B[i]; (i < (M - 1) && k < irp_B[i + 1]) || (i >= M - 1 && k < nz); k++)
            {
                if (ja_A[j] == ja_B[k])
                {
                    found++;
                    if (found > 1) //It means that there are at least two elements with the same column for a fixed row
                    {
                        printf("The two conversions are different (found > 1)\n");
                        return 1;
                    }
                    if (as_A[j] != as_B[k]) //The two elements with same row and column index are different 
                    {
                        printf("The two conversions are different\n");
                        return 1;
                    }
                }
            }
            if (found == 0)
            {
                printf("The two conversions are different (found = 0)\n");
                return 1;
            }
        }
    }

    /**
     * Starting freeing memory
    */

    if (I != NULL)
        free(I);
    if (J != NULL)
        free(J);
    if (val != NULL)
        free(val);

    free_CSR_data_structures(as_A, ja_A, irp_A);
    free_CSR_data_structures(as_B, ja_B, irp_B);

    if (nz_per_row_A != NULL)
        free(nz_per_row_A);
    if (nz_per_row_B != NULL)
        free(nz_per_row_B);

    printf("Same CSR conversions\n");

    return 0;
}
#endif

/**
 * check_correctness - Compare the two matrices returned by the serial and parallel algorithm
 * @param M: Number of rows of the resulting matrix
 * @param K: Number of columns of the resulting matrix
 * @param y_serial: Resulting matrix returned by the serial product algorithm
 * @param y_parallel: resulting matrix returned by the parallel product algorithm
 * 
 */
#ifdef OPENMP
void check_correctness(int M, int K, double ** y_serial, double ** y_parallel)
#elif CUDA
void check_correctness(int M, int K, double ** y_serial, double * y_parallel)
#endif
{

    double abs_err = 0.0;
    double rel_err = 0.0;

    for (int i = 0; i < M; i++)
    {
        for (int z = 0; z < K; z++)
        {
        /**
         * Computing the absolute error and relative error for each element in y
        */
#ifdef CUDA
            double max_abs = max(fabs(y_serial[i][z]), fabs(y_parallel[i * K + z]));
            abs_err = fabs(y_serial[i][z] - y_parallel[i * K + z]);
#elif OPENMP
            double max_abs = max(fabs(y_serial[i][z]), fabs(y_parallel[i][z]));
            abs_err = fabs(y_serial[i][z] - y_parallel[i][z]);
#endif  
         if (max_abs == 0.0) max_abs = 1.0;

        rel_err = max(rel_err, abs_err / max_abs);
    
        
        }
    }
     /**
         * Checking if the two resulting matrixes are equal or not. 
         * TOLLERANZA (2.22e^-16 is the threshold) is the IEEE unit roundoff for a double 
        */
    printf("I due prodotti matrice-matrice hanno un max relative error pari a %.20lf (2.22e^-16 is the threshold).\n", rel_err);

  
}