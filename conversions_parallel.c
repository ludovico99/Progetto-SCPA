#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include "header.h"

int coo_to_ellpack_parallel(int M, int N, int nz, int *I, int *J, double *val, double ***values, int ***col_indices, int nthread)
{
    int max_nz_per_row = 0;
    int max_so_far = 0;
    int nz_in_row;
    int offset;
    int chunk_size;

    printf("ELLPACK parallel started...\n");

    chunk_size = compute_chunk_size(M, nthread);

#pragma omp parallel shared(I, J, val, max_so_far, M, nz, chunk_size) firstprivate(max_nz_per_row) private(nz_in_row) num_threads(nthread) default(none)
    {
// Calcola il massimo numero di elementi non nulli in una riga
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

    // Alloca memoria per gli array ELLPACK
    if (val != NULL)
    {
        (*values) = (double **)malloc(M * sizeof(double *));
        if (*values == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }

        for (int k = 0; k < M; k++)
        {
            (*values)[k] = (double *)malloc(max_so_far * sizeof(double));
            if ((*values)[k] == NULL)
            {
                printf("Errore malloc\n");
                exit(1);
            }
        }
    }

    *col_indices = (int **)malloc(M * sizeof(int *));
    if (*col_indices == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }
    for (int k = 0; k < M; k++)
    {

        (*col_indices)[k] = (int *)malloc(max_so_far * sizeof(int));
        if ((*col_indices)[k] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
    }

    printf("Malloc for ELLPACK data structures completed\n");

    // Riempie gli array ELLPACK con i valori e gli indici di colonna corrispondenti
    int counter = 0;

#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, M, values, col_indices, I, val, J, nz, max_so_far, counter) num_threads(nthread) private(offset) default(none)
    for (int i = 0; i < M; i++)
    {

        offset = 0;
        for (int j = 0; j < nz; j++)
        {
            /*warning: In questa conversione parallela gli array values e col_indices in diverse iterazioni rimangono invariati
             * poichè il thread i-esimo va a modificare la componente (i, offset), entrambe variabili private per i. Di conseguenza,
             * il risultato del prodotto matrice-vettore, nonostante non sia commutativo in aritmetica floating point, rimane invariato
             * in diverse esecuzioni.
             */
            if (I[j] == i)
            {
                if (val != NULL)
                    (*values)[i][offset] = val[j];
                (*col_indices)[i][offset] = J[j];
                offset++;
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
                (*col_indices)[i][k] = -1;
        }
#pragma omp atomic
        counter++;
        printf("%d M completed\n", counter);
    }

    printf("ELLPACK parallel completed...\n");

    // for (int j = 0; j < M; j++)
    // {
    //     for(int k = 0; k < max_so_far; k++)
    //     {
    //        AUDIT printf("ELLPACK VALUE: %.66lf - COL: %d\n", (*values)[j][k], (*col_indices)[j][k]);
    //     }
    // }

    return max_so_far;
}

int *coo_to_CSR_parallel(int M, int N, int nz, int *I, int *J, double *val, double **as, int **ja, int **irp, int nthread)
{
    int *occ;
    int *sum_occ;
    int chunk_size = 0;
    int not_empty = 0;
    int num_first_nz_current_row;
    int my_offset;

    printf("Starting parallel CSR conversion ...\n");

    occ = (int *)calloc(M, sizeof(int));
    if (occ == NULL)
    {
        printf("Errore malloc per occ\n");
        exit(1);
    }

    sum_occ = (int *)calloc(M, sizeof(int));
    if (sum_occ == NULL)
    {
        printf("Errore malloc per sum_occ\n");
        exit(1);
    }

    // Alloca memoria per gli array CSR
    if (val != NULL)
    {
        *as = (double *)malloc(nz * sizeof(double));
        if (*as == NULL)
        {
            printf("Errore malloc per as\n");
            exit(1);
        }
    }

    *ja = (int *)malloc(nz * sizeof(int));
    if (*ja == NULL)
    {
        printf("Errore malloc per ja\n");
        exit(1);
    }

    *irp = (int *)malloc(M * sizeof(int));
    if (*irp == NULL)
    {
        printf("Errore malloc per ja\n");
        exit(1);
    }

    memset(*irp, -1, sizeof(int) * M);

    for (int i = 0; i < nz; i++)
    {
        occ[I[i]]++;
    }

    for (int i = 0; i < M; i++)
    {
        if (i == 0)
            sum_occ[i] = occ[0];
        else
            sum_occ[i] = sum_occ[i - 1] + occ[i];
    }

    printf("Filling CSR data structure ... \n");

    // Riempie gli array CSR

    chunk_size = compute_chunk_size(M, nthread);

#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, M, nz, ja, as, irp, val, I, J, sum_occ) private(my_offset, not_empty, num_first_nz_current_row) default(none)

    /*warning: In questa conversione parallela gli array as e ja in diverse iterazioni rimangono invariati
     * poichè il thread i-esimo va a modificare la componente (my_offset), variabile privata per i. Di conseguenza,
     * il risultato del prodotto matrice-vettore, nonostante non sia commutativo in aritmetica floating point, rimane invariato
     * in diverse esecuzioni.
     */

    for (int i = 0; i < M; i++)
    {
        not_empty = 0;
        num_first_nz_current_row;

        if (i == 0)
        {
            num_first_nz_current_row = 0;
        }
        else
            num_first_nz_current_row = sum_occ[i - 1];

        (*irp)[i] = num_first_nz_current_row;
        my_offset = num_first_nz_current_row;

        for (int j = 0; j < nz; j++)
        {
            if (I[j] == i)
            {
                if (val != NULL)
                    (*as)[my_offset] = val[j];
                (*ja)[my_offset] = J[j];
                my_offset++;
                not_empty = 1;
            }
        }
        if (!not_empty)
            (*irp)[i] = -1;
    }

    if (sum_occ != NULL )free(sum_occ);

    #ifndef CHECK_CONVERSION 
        printf("Freeing COO data structures...\n");
        if (I != NULL) free(I);
        if (J != NULL) free(J);
        if (val != NULL) free(val);
    #endif

    printf("Completed parallel CSR conversion ...\n");

    return occ;
}

int *coo_to_CSR_parallel_optimization(int M, int N, int nz, int *I, int *J, double *val, double **as, int **ja, int **irp, int nthread)
{
    int *nz_per_row = NULL;
    int chunk_size = 0;

    printf("Starting parallel CSR conversion ...\n");
    nz_per_row = (int *)calloc(M, sizeof(int));

    if (nz_per_row == NULL)
    {
        printf("Errore malloc per nz_per_row\n");
        exit(1);
    }

    printf("Counting number of non-zero entries in each row...\n");

    chunk_size = compute_chunk_size(nz, nthread);

#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, I, nz, nz_per_row, stdout) default(none)

    for (int i = 0; i < nz; i++)
    {
#pragma omp atomic
        nz_per_row[I[i]]++;
    }

    printf("Allocating memory ...\n");

    // Alloca memoria per gli array CSR
    if (val != NULL)
    {
        *as = (double *)malloc(nz * sizeof(double));
        if (*as == NULL)
        {
            printf("Errore malloc per as\n");
            exit(1);
        }
    }

    *ja = (int *)malloc(nz * sizeof(int));
    if (*ja == NULL)
    {
        printf("Errore malloc per ja\n");
        exit(1);
    }

    *irp = (int *)malloc(M * sizeof(int));
    if (*irp == NULL)
    {
        printf("Errore malloc per ja\n");
        exit(1);
    }

    memset(*irp, -1, sizeof(int) * M);

    printf("Filling CSR data structure ... \n");
    // Riempie gli array CSR

    if (nz_per_row[0] == 0)
    {
        (*irp)[0] = -1;
        (*irp)[1] = 0;
    }
    else
    {
        (*irp)[0] = 0;
        (*irp)[1] = (*irp)[0] + nz_per_row[0];
    }
    for (int i = 1; i < M - 1; i++)
    {
        (*irp)[i + 1] = (*irp)[i] + nz_per_row[i];
    }

    int offset = 0;
    int row;
    int idx;
#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, nz, irp, ja, as, val, offset, I, J) private(row, idx) default(none)

    for (int i = 0; i < nz; i++)
    {
         /*warning: In questa conversione parallela gli array irp e as in diverse iterazioni possono cambiare
         * poichè il thread i-esimo e un altro thread j, in concorrenza, possono essere responsabili di diversi elementi (I[i], J[i], val [i] <-->
         * I[j] == I[i], J[j], val [j])della stessa riga (I[i] == I[j]).
         * La variabile idx, che è l'indice all'interno dell'array as in corrispondenza del quale assegnare elemento corrente,
         * è uguale per entrambi. E' possibile quindi, che in diverse esecuzioni, l'elemento x sia posizionato in posizione (idx) dal thread i
         * oppure che l'elemento y sia posizionato in (idx) dal thread j. "Vince" chi accede prima al blocco critical.
         * Di conseguenza, il risultato del prodotto matrice-vettore, poichè è commutativo in aritmetica floating point, cambia in diverse esecuzioni.
         */
        row = I[i];
#pragma omp critical(CSR_optimization)
        {
            idx = (*irp)[row];
            (*ja)[idx] = J[i];
            if (val != NULL)
                (*as)[idx] = val[i];
            (*irp)[row]++;
        }
    }

    // Reset row pointers
    for (int i = M - 1; i > 0; i--)
    {
        (*irp)[i] = (*irp)[i - 1];
    }

    if (nz_per_row[0] == 0)
    {
        (*irp)[0] = -1;
        (*irp)[1] = 0;
    }
    else
        (*irp)[0] = 0;

    #ifndef CHECK_CONVERSION 
        printf("Freeing COO data structures...\n");
        if (I != NULL) free(I);
        if (J != NULL) free(J);
        if (val != NULL) free(val);
    #endif

    printf("Completed parallel CSR conversion ...\n");

    return nz_per_row;
}

int *coo_to_ellpack_no_zero_padding_parallel(int M, int N, int nz, int *I, int *J, double *val, double ***values, int ***col_indices, int nthread)
{
    int offset;
    int chunk_size;
    int *nz_per_row = NULL;

    printf("ELLPACK parallel started...\n");

    chunk_size = compute_chunk_size(nz, nthread);

    nz_per_row = (int *)calloc(M, sizeof(int));

    if (nz_per_row == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }

#pragma omp parallel for schedule(static, chunk_size) shared(I, nz, chunk_size, nz_per_row) num_threads(nthread) default(none)
    // Calcola il numero di elementi non nulli per riga
    for (int i = 0; i < nz; i++)
    {
#pragma omp atomic
        nz_per_row[I[i]]++;
    }

    printf("Number of zeroes in rows computed\n");

    // Alloca memoria per gli array ELLPACK
    if (val != NULL)
    {
        (*values) = (double **)malloc(M * sizeof(double *));
        if (*values == NULL)
        {
            printf("Errore malloc for values\n");
            exit(1);
        }

        for (int k = 0; k < M; k++)
        {

            if (nz_per_row[k] != 0)
            {
                (*values)[k] = (double *)malloc(nz_per_row[k] * sizeof(double));
                if ((*values)[k] == NULL)
                {
                    printf("Errore malloc for values[k] for row %d nz_per_row[] %d\n", k, nz_per_row[k]);
                    exit(1);
                }
            }
            else
                (*values)[k] = NULL;
        }
    }

    *col_indices = (int **)malloc(M * sizeof(int *));
    if (*col_indices == NULL)
    {
        printf("Errore malloc for col_indices\n");
        exit(1);
    }
    for (int k = 0; k < M; k++)
    {

        if (nz_per_row[k] != 0)
            (*col_indices)[k] = (int *)malloc(nz_per_row[k] * sizeof(int));
        else
            (*col_indices)[k] = NULL;
        if ((*col_indices)[k] == NULL)
        {
            printf("Errore malloc for col_indices[k]\n");
            exit(1);
        }
    }

    printf("Malloc for ELLPACK data structures completed\n");

#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, M, values, col_indices, I, val, J, nz, nz_per_row) num_threads(nthread) private(offset) default(none)
    for (int i = 0; i < M; i++)
    {

        offset = 0;
        if (nz_per_row[i] == 0)
            continue;

        for (int j = 0; j < nz; j++)
        {
            /*warning: In questa conversione parallela gli array 2D values e col_indices in diverse iterazioni rimangono invariati
             * poichè il thread i-esimo va a modificare la componente (i, offset), entrambe variabili private per i. Di conseguenza,
             * il risultato del prodotto matrice-vettore, nonostante non sia commutativo in aritmetica floating point, rimane invariato
             * in diverse esecuzioni.
             */
            if (I[j] == i)
            {
                if (val != NULL)
                    (*values)[i][offset] = val[j];
                (*col_indices)[i][offset] = J[j];
                offset++;
            }
        }
        if (offset != nz_per_row[i])
        {
            printf("offset maggiore di nz_per_row");
            exit(1);
        }
    }

    #ifndef CHECK_CONVERSION 
        printf("Freeing COO data structures...\n");
        if (I != NULL) free(I);
        if (J != NULL) free(J);
        if (val != NULL) free(val);
    #endif

    printf("ELLPACK parallel completed...\n");

    return nz_per_row;
}

int *coo_to_ellpack_no_zero_padding_parallel_optimization(int M, int N, int nz, int *I, int *J, double *val, double ***values, int ***col_indices, int nthread)
{
    int offset;
    int chunk_size;
    int *nz_per_row = NULL;

    printf("ELLPACK parallel started...\n");

    int *curr_idx_per_row = (int *)calloc(M, sizeof(int));

    if (curr_idx_per_row == NULL)
    {
        printf("Errore malloc per curr_idx_per_row \n");
        exit(1);
    }

    chunk_size = compute_chunk_size(nz, nthread);

    nz_per_row = (int *)calloc(M, sizeof(int));

    if (nz_per_row == NULL)
    {
        printf("Errore malloc per nz_per_row \n");
        exit(1);
    }

#pragma omp parallel for schedule(static, chunk_size) shared(I, nz, chunk_size, nz_per_row) num_threads(nthread) default(none)
    // Calcola il numero di elementi non nulli per riga
    for (int i = 0; i < nz; i++)
    {
#pragma omp atomic
        nz_per_row[I[i]]++;
    }

    printf("Number of zeroes in rows computed\n");

    // Alloca memoria per gli array ELLPACK
    if (val != NULL)
    {
        (*values) = (double **)malloc(M * sizeof(double *));
        if (*values == NULL)
        {
            printf("Errore malloc for values\n");
            exit(1);
        }

        for (int k = 0; k < M; k++)
        {

            if (nz_per_row[k] != 0)
            {
                (*values)[k] = (double *)malloc(nz_per_row[k] * sizeof(double));
                if ((*values)[k] == NULL)
                {
                    printf("Errore malloc for values[k]\n");
                    exit(1);
                }
            }
            else
                (*values)[k] = NULL;
        }
    }

    *col_indices = (int **)malloc(M * sizeof(int *));
    if (*col_indices == NULL)
    {
        printf("Errore malloc for col_indices\n");
        exit(1);
    }
    for (int k = 0; k < M; k++)
    {

        if (nz_per_row[k] != 0)
        {
            (*col_indices)[k] = (int *)malloc(nz_per_row[k] * sizeof(int));
            if ((*col_indices)[k] == NULL)
            {
                printf("Errore malloc for col_indices[k]\n");
                exit(1);
            }
        }
        else
            (*col_indices)[k] = NULL;
    }

    printf("Malloc for ELLPACK data structures completed\n");
    fflush(stdout);

    int row_elem;
    int col_elem;
    double val_elem;
    int k;
#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, values, col_indices, I, val, J, nz, nz_per_row, curr_idx_per_row) num_threads(nthread) private(row_elem, k, col_elem, val_elem) default(none)
    for (int i = 0; i < nz; i++)
    {

        /*warning: In questa conversione parallela gli array 2D values e col_indices in diverse iterazioni possono cambiare
         * poichè il thread i-esimo e un altro thread j, in concorrenza, possono essere responsabili di diversi elementi della stessa riga.
         * La variabile k, che è l'indice corrente all'interno di quella riga in corrispondenza del quale assegnare elemento corrente,
         * è uguale per entrambi. E' possibile quindi, che in diverse esecuzioni, l'elemento x sia posizionato in posizione (riga di x, k) dal thread i
         * oppure che l'elemento y sia posizionato in (riga di y, k) dal thread j. NB: riga di x == riga di y
         * Di conseguenza, il risultato del prodotto matrice-vettore, poichè è commutativo in aritmetica floating point, cambia in diverse esecuzioni.
         */
        row_elem = I[i];
        col_elem = J[i];
        val_elem = val[i];
#pragma omp critical(ELLPACK_optimization)
        {

            k = curr_idx_per_row[row_elem];

            if (k >= nz_per_row[row_elem])
            {
                printf("offset maggiore di nz_per_row");
                exit(1);
            }

            if (k < nz_per_row[row_elem])
            {

                if (val != NULL)
                    (*values)[row_elem][k] = val[i];
                (*col_indices)[row_elem][k] = J[i];
                curr_idx_per_row[row_elem]++;
            }
        }
    }

    if (curr_idx_per_row != NULL) free(curr_idx_per_row);

    #ifndef CHECK_CONVERSION 
        printf("Freeing COO data structures...\n");
        if (I != NULL) free(I);
        if (J != NULL) free(J);
        if (val != NULL) free(val);
    #endif

    printf("ELLPACK parallel completed...\n");

    return nz_per_row;
}
