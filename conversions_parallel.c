#include <omp.h>
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
 * coo_to_CSR_parallel - Non-time-optimized parallel version for converting a sparse matrix from COO format to CSR format
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 * @param as: Pointer to coefficient vector
 * @param ja: Pointer to the column index vector
 * @param irp: Pointer to the vector of the start index of each row
 * @param nthread: number of processors available to the device.
 */

int *coo_to_CSR_parallel(int M, int N, int nz, int *I, int *J, double *val, double **as, int **ja, int **irp, int nthread)
{
    int *nz_per_row;
    int chunk_size = 0;

    printf("Starting parallel CSR conversion ...\n");

    all_zeroes_memory_allocation(int, M, nz_per_row);

    /**
     * Allocating memory for the CSR vectors
     */

    if (val != NULL)
    {
        memory_allocation(double, nz, *as);
    }

    all_zeroes_memory_allocation(int, nz, *ja);

    memory_allocation(int, M, *irp);
    memset(*irp, -1, sizeof(int) * M);

    printf("Counting number of non-zero entries in each row...\n");

    chunk_size = compute_chunk_size(nz, nthread);

    /**
     * Computing the number of not zero for each row
     */
#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, I, nz, nz_per_row, stdout) default(none)

    for (int i = 0; i < nz; i++)
    {
#pragma omp atomic
        nz_per_row[I[i]]++;
    }

    /**
     * Computing, given row i, the sum of non-zeros up to the i-th row
     */

    (*irp)[0] = 0;
    for (int i = 0; i < M - 1; i++)
    {
        (*irp)[i + 1] = (*irp)[i] + nz_per_row[i]; // By defition irp[i] is the index of the first not zero for each row
    }

    printf("Filling CSR data structure ... \n");

    /**
     * Filling the CSR data structures
     */

#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, M, nz, ja, as, irp, val, I, J) default(none)
    /**
     * The threads divide the rows equally. Each thread has at least chunk_size ierations
     */

    /**
     * WARNING: In questa conversione parallela gli array as e ja in diverse iterazioni rimangono invariati
     * poichè il thread i-esimo va a modificare la componente (my_offset), variabile privata per i. Di conseguenza,
     * il risultato del prodotto matrice-vettore, nonostante non sia commutativo in aritmetica floating point, rimane invariato
     * in diverse esecuzioni.
     */

    for (int i = 0; i < M; i++)
    {
        int not_empty = 0;
        int num_first_nz_current_row; // It's the index in the vectors as and ja of the first not zero for each row
        int my_offset = (*irp)[i];    // my_offset is the index of the first not zero for each row too

        for (int j = 0; j < nz; j++)
        {
            if (I[j] == i)
            {
                if (val != NULL)
                    (*as)[my_offset] = val[j];
                (*ja)[my_offset] = J[j];
                my_offset++; // Throught the variable my_offset the thread assigned to the i-th row iterates on as and ja
                not_empty = 1;
            }
        }
        if (!not_empty)
            (*irp)[i] = -1;
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

    printf("Completed parallel CSR conversion ...\n");

    return nz_per_row;
}

/**
 * coo_to_CSR_parallel_optimization - Time-optimized parallel version for converting a sparse matrix from COO format to CSR format
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 * @param as: Pointer to coefficient vector
 * @param ja: Pointer to the column index vector
 * @param irp: Pointer to the vector of the start index of each row
 * @param nthread: number of processors available to the device.
 */
int *coo_to_CSR_parallel_optimization(int M, int N, int nz, int *I, int *J, double *val, double **as, int **ja, int **irp, int nthread)
{
    int *nz_per_row = NULL;
    int chunk_size = 0;
    int offset = 0;

    printf("Starting parallel CSR conversion ...\n");

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

    chunk_size = compute_chunk_size(nz, nthread);

    printf("Counting number of non-zero entries in each row...\n");

    /**
     * Computing the number of not zero for each row
     */
#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, I, nz, nz_per_row) default(none)

    for (int i = 0; i < nz; i++)
    {
#pragma omp atomic
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

#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, nz, irp, ja, as, val, offset, I, J) default(none)
    /**
     * Threads share the iteration space equally. Each thread has at least chunk_size ierations
     */
    for (int i = 0; i < nz; i++)
    {
        /**
         * WARNING: In questa conversione parallela gli array irp e as in diverse iterazioni possono cambiare
         * poichè il thread i-esimo e un altro thread j, in concorrenza, possono essere responsabili di diversi elementi (I[i], J[i], val [i] <-->
         * I[j] == I[i], J[j], val [j])della stessa riga (I[i] == I[j]).
         * La variabile idx, che è l'indice all'interno dell'array as in corrispondenza del quale assegnare elemento corrente,
         * è uguale per entrambi. E' possibile quindi, che in diverse esecuzioni, l'elemento x sia posizionato in posizione (idx) dal thread i
         * oppure che l'elemento y sia posizionato in (idx) dal thread j. "Vince" chi accede prima al blnz_per_rowo critical.
         * Di conseguenza, il risultato del prodotto matrice-vettore, poichè è commutativo in aritmetica floating point, cambia in diverse esecuzioni.
         */

        int row = I[i];
#pragma omp critical(CSR_optimization)
        {
            int idx = (*irp)[row];
            (*ja)[idx] = J[i];
            if (val != NULL)
                (*as)[idx] = val[i];
            (*irp)[row]++;
        }
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

#ifndef CHECK_CONVERSION
    printf("Freeing COO data structures...\n");
    if (I != NULL)
        free(I);
    if (J != NULL)
        free(J);
    if (val != NULL)
        free(val);
#endif

    printf("Completed parallel CSR conversion ...\n");

    return nz_per_row;
}

/*------------------------------------------------------- ELLPACK ---------------------------------------------------------------------*/

/**
 * coo_to_ellpack_parallel - Parallel version with 0x0 padding for converting a sparse matrix from COO format to ELLPACK format
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 * @param values: Pointer to the 2D array of coefficients
 * @param col_indices: Pointer to the 2D array of column indexes
 * @param nthread: Number of processors available to the device.
 */

int coo_to_ellpack_parallel(int M, int N, int nz, int *I, int *J, double *val, double ***values, int ***col_indices, int nthread)
{
    int max_nz_per_row = 0;
    int max_so_far = 0;
    int nz_in_row;
    int chunk_size;

    printf("ELLPACK parallel started...\n");

    chunk_size = compute_chunk_size(M, nthread);

    /**
     * Allocating memory for the ELLPACK 2D data structures
     */
    if (val != NULL)
    {
        memory_allocation(double *, M, *values);

        for (int k = 0; k < M; k++)
        {
            memory_allocation(double, max_so_far, (*values)[k]);
        }
    }

    memory_allocation(int *, M, *col_indices);

    for (int k = 0; k < M; k++)
    {
        memory_allocation(int, max_so_far, (*col_indices)[k]);
    }

    printf("Malloc for ELLPACK data structures completed\n");

    /**
     * Calculates the maximum number of non-zero elements across all rows
     */
#pragma omp parallel shared(I, J, val, max_so_far, M, nz, chunk_size) firstprivate(max_nz_per_row) private(nz_in_row) num_threads(nthread) default(none)
    {

#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < M; i++)
        {
            nz_in_row = 0;
            for (int j = 0; j < nz; j++)
            {
                if (I[j] == i)
                    nz_in_row++;
            }
            if (nz_in_row > max_nz_per_row)
                max_nz_per_row = nz_in_row;
        }

#pragma omp critical
        if (max_nz_per_row > max_so_far)
            max_so_far = max_nz_per_row;
    }

    printf("MAX_NZ_PER_ROW is %d\n", max_so_far);

    /**
     * Fills ELLPACK arrays with values and corresponding column indexes
     */

#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, M, values, col_indices, I, val, J, nz, max_so_far) num_threads(nthread) default(none)
    for (int i = 0; i < M; i++)
    {

        int offset = 0;
        for (int j = 0; j < nz; j++)
        {
            /**
             * WARNING: In questa conversione parallela gli array values e col_indices in diverse iterazioni rimangono invariati
             * poichè il thread i-esimo va a modificare la componente (i, offset), entrambe variabili private per i. Di conseguenza,
             * il risultato del prodotto matrice-vettore, nonostante non sia commutativo in aritmetica floating point, rimane invariato
             * in diverse esecuzioni.
             */
            if (I[j] == i)
            {
                if (val != NULL)
                    (*values)[i][offset] = val[j];
                (*col_indices)[i][offset] = J[j];
                offset++; // Throught the variable offset the thread assigned to the i-th row iterates on the arrays pointed by values[i] and col_indices[i]
            }

            if (offset > max_so_far)
            {
                printf("offset maggiore di max_so_far");
                exit(1);
            }
        }

        for (int k = offset; k < max_so_far; k++)
        {
            if (val != NULL)
                (*values)[i][k] = 0.0;

            if (offset != 0)
                (*col_indices)[i][k] = (*col_indices)[i][offset - 1];
            else
                (*col_indices)[i][k] = -1; // The i-th row has 0 not zero elements
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

    printf("ELLPACK parallel completed...\n");

    return max_so_far;
}

/**
 * coo_to_ellpack_no_zero_padding_parallel - Non-time-optimized parallel version without 0x0 padding for converting a sparse matrix from COO format to ELLPACK format
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 * @param values: Pointer to the 2D array of coefficients
 * @param col_indices: Pointer to the 2D array of column indexes
 * @param nthread: number of processors available to the device.
 */

int *coo_to_ellpack_no_zero_padding_parallel(int M, int N, int nz, int *I, int *J, double *val, double ***values, int ***col_indices, int nthread)
{
    int offset;
    int chunk_size;
    int *nz_per_row = NULL;

    printf("ELLPACK parallel started...\n");

    chunk_size = compute_chunk_size(nz, nthread);

    all_zeroes_memory_allocation(int, M, nz_per_row);

    /**
     * Calculates the number of non-zero elements per row
     */

#pragma omp parallel for schedule(static, chunk_size) shared(I, nz, chunk_size, nz_per_row) num_threads(nthread) default(none)
    for (int i = 0; i < nz; i++)
    {
#pragma omp atomic
        nz_per_row[I[i]]++;
    }

    printf("Number of zeroes per rows computed\n");

    /**
     * Allocating memory for the ELLPACK 2D data structures
     */

    if (val != NULL)
    {
        memory_allocation(double *, M, *values);

        for (int k = 0; k < M; k++)
        {

            if (nz_per_row[k] != 0)
            {
                memory_allocation(double, nz_per_row[k], (*values)[k]);
            }
            else
                (*values)[k] = NULL;
        }
    }

    memory_allocation(int *, M, *col_indices);
    for (int k = 0; k < M; k++)
    {

        if (nz_per_row[k] != 0)
        {
            memory_allocation(int, nz_per_row[k], (*col_indices)[k]);
        }

        else
            (*col_indices)[k] = NULL;
    }

    printf("Malloc for ELLPACK data structures completed\n");

    /**
     * Fills ELLPACK arrays with values and corresponding column indexes
     */

#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, M, values, col_indices, I, val, J, nz, nz_per_row) num_threads(nthread) private(offset) default(none)
    for (int i = 0; i < M; i++)
    {

        offset = 0;
        if (nz_per_row[i] == 0)
            continue;

        for (int j = 0; j < nz; j++)
        {
            /**
             * WARNING: In questa conversione parallela gli array 2D values e col_indices in diverse iterazioni rimangono invariati
             * poichè il thread i-esimo va a modificare la componente (i, offset), entrambe variabili private per i. Di conseguenza,
             * il risultato del prodotto matrice-vettore, nonostante non sia commutativo in aritmetica floating point, rimane invariato
             * in diverse esecuzioni.
             */
            if (I[j] == i)
            {
                if (val != NULL)
                    (*values)[i][offset] = val[j];
                (*col_indices)[i][offset] = J[j];
                offset++; // Throught the variable offset the thread assigned to the i-th row iterates on the arrays pointed by values[i] and col_indices[i]
            }
        }
        /**
         * Consistency check: offset should be equal to nz_per_row[i] for the i-th row
         */
        if (offset != nz_per_row[i])
        {
            printf("offset maggiore di nz_per_row");
            exit(1);
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

    printf("ELLPACK parallel completed...\n");

    return nz_per_row;
}

/**
 * coo_to_ellpack_no_zero_padding_parallel_optimization - Time-optimized parallel version without 0x0 padding for converting a sparse matrix from COO format to ELLPACK format
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 * @param values: Pointer to the 2D array of coefficients
 * @param col_indices: Pointer to the 2D array of column indexes
 * @param nthread: number of processors available to the device.
 */
int *coo_to_ellpack_no_zero_padding_parallel_optimization(int M, int N, int nz, int *I, int *J, double *val, double ***values, int ***col_indices, int nthread)
{
    int offset;
    int chunk_size;
    int *nz_per_row = NULL;
    int *curr_idx_per_row = NULL; // It memorizes the actual index per row
    int row_elem;
    int col_elem;
    double val_elem;
    int k;

    printf("ELLPACK parallel started...\n");

    chunk_size = compute_chunk_size(nz, nthread);

    all_zeroes_memory_allocation(int, M, curr_idx_per_row);
    all_zeroes_memory_allocation(int, M, nz_per_row);

    /**
     * Calculates the number of non-zero elements per row
     */

#pragma omp parallel for schedule(static, chunk_size) shared(I, nz, chunk_size, nz_per_row) num_threads(nthread) default(none)
    for (int i = 0; i < nz; i++)
    {
#pragma omp atomic
        nz_per_row[I[i]]++;
    }

    printf("Number of zeroes in rows computed\n");

    /**
     * Allocating memory for the ELLPACK 2D data structures
     */

    if (val != NULL)
    {
        memory_allocation(double *, M, *values);

        for (int k = 0; k < M; k++)
        {

            if (nz_per_row[k] != 0)
            {
                memory_allocation(double, nz_per_row[k], (*values)[k]);
            }
            else
                (*values)[k] = NULL;
        }
    }

    memory_allocation(int *, M, *col_indices);
    for (int k = 0; k < M; k++)
    {

        if (nz_per_row[k] != 0)
        {
            memory_allocation(int, nz_per_row[k], (*col_indices)[k]);
        }

        else
            (*col_indices)[k] = NULL;
    }

    printf("Malloc for ELLPACK data structures completed\n");

    /**
     * Fills ELLPACK arrays with values and corresponding column indexes
     */

#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, values, col_indices, I, val, J, nz, nz_per_row, curr_idx_per_row) num_threads(nthread) default(none)
    for (int i = 0; i < nz; i++)
    {
        /**
         * WARNING: In questa conversione parallela gli array 2D values e col_indices in diverse iterazioni possono cambiare
         * poichè il thread i-esimo e un altro thread j, in concorrenza, possono essere responsabili di diversi elementi della stessa riga.
         * La variabile k, che è l'indice corrente all'interno di quella riga in corrispondenza del quale assegnare elemento corrente,
         * è uguale per entrambi. E' possibile quindi, che in diverse esecuzioni, l'elemento x sia posizionato in posizione (riga di x, k) dal thread i
         * oppure che l'elemento y sia posizionato in (riga di y, k) dal thread j. NB: riga di x == riga di y
         * Di conseguenza, il risultato del prodotto matrice-vettore, poichè è commutativo in aritmetica floating point, cambia in diverse esecuzioni.
         */
        int row_elem = I[i];
        int col_elem = J[i];
        double val_elem = val[i];

#pragma omp critical(ELLPACK_optimization)
        {

            int k = curr_idx_per_row[row_elem];

            /**
             * Consistency check: offset should be greater or equal than nz_per_row[row_elem] for the i-th row
             */

            if (k >= nz_per_row[row_elem])
            {
                printf("offset maggiore di nz_per_row");
                exit(1);
            }

            if (val != NULL)
                (*values)[row_elem][k] = val[i];

            (*col_indices)[row_elem][k] = J[i];
            curr_idx_per_row[row_elem]++;
        }
    }

    /**
     * Freeing memory
     */

    if (curr_idx_per_row != NULL)
        free(curr_idx_per_row);

#ifndef CHECK_CONVERSION
    printf("Freeing COO data structures...\n");
    if (I != NULL)
        free(I);
    if (J != NULL)
        free(J);
    if (val != NULL)
        free(val);
#endif

    printf("ELLPACK parallel completed...\n");

    return nz_per_row;
}
