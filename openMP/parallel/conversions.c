#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include "../header.h"

int coo_to_ellpack_parallel(int rows, int columns, int nz, int *I, int *J, double *val, double ***values, int ***col_indices)
{
    int max_nz_per_row = 0;
    int max_so_far = 0;
    int nthreads;
    int nz_in_row;
    int offset;
    int chunk_size;

    nthreads = omp_get_num_procs();

    printf("ELLPACK parallel started...\n");

    if (rows % nthreads == 0)
    {
        chunk_size = rows / nthreads;
    }
    else
    {
        chunk_size = (rows / nthreads) + 1;
    }

#pragma omp parallel shared(I, J, val, max_so_far, rows, nz, chunk_size) firstprivate(max_nz_per_row) private(nz_in_row) num_threads(nthreads) default(none)
    {
// Calcola il massimo numero di elementi non nulli in una riga
#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < rows; i++)
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
        (*values) = (double **)malloc(rows * sizeof(double *));
        if (*values == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }

        for (int k = 0; k < rows; k++)
        {
            (*values)[k] = (double *)malloc(max_so_far * sizeof(double));
            if ((*values)[k] == NULL)
            {
                printf("Errore malloc\n");
                exit(1);
            }
        }
    }

    *col_indices = (int **)malloc(rows * sizeof(int *));
    if (*col_indices == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }
    for (int k = 0; k < rows; k++)
    {

        (*col_indices)[k] = (int *)malloc(max_so_far * sizeof(int));
        if ((*col_indices)[k] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
    }

    printf("Malloc for ELLPACK data structures completed\n");
    fflush(stdout);
    // Riempie gli array ELLPACK con i valori e gli indici di colonna corrispondenti
    int counter = 0;
#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, rows, values, col_indices, I, val, J, nz, max_so_far, counter) num_threads(nthreads) private(offset) default(none)
    for (int i = 0; i < rows; i++)
    {

        offset = 0;
        for (int j = 0; j < nz; j++)
        {
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
        printf("%d rows completed\n", counter);
    }

    printf("ELLPACK parallel completed...\n");

    // for (int j = 0; j < rows; j++)
    // {
    //     for(int k = 0; k < max_so_far; k++)
    //     {
    //        AUDIT printf("ELLPACK VALUE: %.66lf - COL: %d\n", (*values)[j][k], (*col_indices)[j][k]);
    //     }
    // }

    return max_so_far;
}

void coo_to_CSR_parallel(int M, int N, int nz, int *I, int *J, double *val, double **as, int **ja, int **irp)
{
    int i, j, k;
    int max_nz_per_row = 0;
    int max_so_far = 0;
    int *end = NULL;
    int *curr = NULL;
    int offset = 0;
    int *occ;
    int *sum_occ;


    printf("Starting parallel CSR conversion ...\n");
    fflush(stdout);

    occ = (int *)malloc(M * sizeof(int));
    if (occ == NULL)
    {
            printf("Errore malloc per occ\n");
            exit(1);
    }

    sum_occ = (int *)malloc(M * sizeof(int));
    if (sum_occ == NULL)
    {
            printf("Errore malloc per sum_occ\n");
            exit(1);
    }

    memset(occ, 0, sizeof(int) * M);
    memset(sum_occ, 0, sizeof(int) * M);

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
        // printf("%d\n", occ[I[i]]);
    }
    int sum = 0;
    for (int i = 0; i < M; i++)
    {
        sum += occ[i];
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
    offset = 0;
    int nthread = omp_get_num_procs();
    int chunk_size = 0;

    if (M % nthread == 0)
    {
        chunk_size = M / nthread;
    }
    else
        chunk_size = M / nthread + 1;

    int not_empty = 0;
    int num_first_nz_current_row;
    int my_offset;

#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, M, nz, ja, as, irp, val, I, J, sum_occ) private(my_offset, not_empty, num_first_nz_current_row) default(none)

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

    free(occ);
    free(sum_occ);

    printf("Completed parallel CSR conversion ...\n");
}

void coo_to_CSR_parallel_optimization(int M, int N, int nz, int *I, int *J, double *val, double **as, int **ja, int **irp)
{
    int * nz_per_row = NULL;
    int nthread = omp_get_num_procs();
    int chunk_size = 0;

    printf("Starting parallel CSR conversion ...\n");
    nz_per_row = (int *)calloc(M, sizeof(int));

    if (nz_per_row == NULL)
    {
        printf("Errore malloc per nz_per_row\n");
        exit(1);
    }

    printf("Counting number of non-zero entries in each row...\n");

    if (nz % nthread == 0)
    {
        chunk_size = nz / nthread;
    }
    else
        chunk_size = nz / nthread + 1;

#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, I, nz, nz_per_row,stdout) default(none)

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

    if (nz_per_row[0] == 0) {
        (*irp)[0] = -1;
        (*irp)[1] = 0;
    }
    else  {
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
        row = I[i];
#pragma omp critical (CSR_optimization)
{
        idx = (*irp)[row];
        (*ja)[idx] = J[i];
        if (val != NULL)(*as)[idx] = val[i];  
        (*irp)[row]++;
}
    }
    

     // Reset row pointers
    for (int i = M - 1; i > 0; i--) {
        (*irp)[i] = (*irp)[i-1];
    }
    
    if (nz_per_row[0] == 0) {
        (*irp)[0] = -1;
        (*irp)[1] = 0;
    }
    else 
        (*irp)[0] = 0;
    
    free(nz_per_row);

    printf("Completed parallel CSR conversion ...\n");
}

int *coo_to_ellpack_no_zero_padding_parallel(int rows, int columns, int nz, int *I, int *J, double *val, double ***values, int ***col_indices)
{
    int nthreads;
    int offset;
    int chunk_size;
    int *nz_per_row = NULL;
    nthreads = omp_get_num_procs();

    printf("ELLPACK parallel started...\n");

    if (nz % nthreads == 0)
    {
        chunk_size = nz / nthreads;
    }
    else
    {
        chunk_size = (nz / nthreads) + 1;
    }

    nz_per_row = (int *)malloc(rows * sizeof(int));

    if (nz_per_row == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }

#pragma omp parallel for schedule(static, chunk_size) shared(I, nz, chunk_size, nz_per_row) num_threads(nthreads) default(none)
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
        (*values) = (double **)malloc(rows * sizeof(double *));
        if (*values == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }

        for (int k = 0; k < rows; k++)
        {

            if (nz_per_row[k] != 0)
                (*values)[k] = (double *)malloc(nz_per_row[k] * sizeof(double));
            else
                (*values)[k] = NULL;
            if ((*values)[k] == NULL)
            {
                printf("Errore malloc\n");
                exit(1);
            }
        }
    }

    *col_indices = (int **)malloc(rows * sizeof(int *));
    if (*col_indices == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }
    for (int k = 0; k < rows; k++)
    {

        if (nz_per_row[k] != 0)
            (*col_indices)[k] = (int *)malloc(nz_per_row[k] * sizeof(int));
        else
            (*col_indices)[k] = NULL;
        if ((*col_indices)[k] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
    }

    printf("Malloc for ELLPACK data structures completed\n");
    fflush(stdout);

#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, rows, values, col_indices, I, val, J, nz, nz_per_row) num_threads(nthreads) private(offset) default(none)
    for (int i = 0; i < rows; i++)
    {

        offset = 0;
        if (nz_per_row[i] == 0)
            continue;

        for (int j = 0; j < nz; j++)
        {
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

    printf("ELLPACK parallel completed...\n");

    // for (int j = 0; j < rows; j++)
    // {
    //     for(int k = 0; k < nz_per_row[j]; k++)
    //     {
    //         printf("ELLPACK VALUE: %.66lf - COL: %d\n", (*values)[j][k], (*col_indices)[j][k]);
    //     }
    // }

    return nz_per_row;
}

int *coo_to_ellpack_no_zero_padding_parallel_optimization(int M, int N, int nz, int *I, int *J, double *val, double ***values, int ***col_indices)
{
    int nthreads;
    int offset;
    int chunk_size;
    int *nz_per_row = NULL;
    nthreads = omp_get_num_procs();

    printf("ELLPACK parallel started...\n");

    int * curr_idx_per_row = (int *)calloc(M, sizeof(int));

    if (curr_idx_per_row == NULL)
    {
        printf("Errore malloc per curr_idx_per_row \n");
        exit(1);
    }

    if (nz % nthreads == 0)
    {
        chunk_size = nz / nthreads;
    }
    else
    {
        chunk_size = (nz / nthreads) + 1;
    }

    nz_per_row = (int *)malloc(M * sizeof(int));

    if (nz_per_row == NULL)
    {
        printf("Errore malloc per nz_per_row \n");
        exit(1);
    }

#pragma omp parallel for schedule(static, chunk_size) shared(I, nz, chunk_size, nz_per_row) num_threads(nthreads) default(none)
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
            printf("Errore malloc\n");
            exit(1);
        }

        for (int k = 0; k < M; k++)
        {

            if (nz_per_row[k] != 0)
                (*values)[k] = (double *)malloc(nz_per_row[k] * sizeof(double));
            else
                (*values)[k] = NULL;
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

        if (nz_per_row[k] != 0)
            (*col_indices)[k] = (int *)malloc(nz_per_row[k] * sizeof(int));
        else
            (*col_indices)[k] = NULL;
        if ((*col_indices)[k] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
    }

    printf("Malloc for ELLPACK data structures completed\n");
    fflush(stdout);

    int row_elem;
    int col_elem;
    double val_elem;
    int k; 
#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, values, col_indices, I, val, J, nz, nz_per_row, curr_idx_per_row) num_threads(nthreads) private(row_elem, k, col_elem, val_elem) default(none)
    for (int i = 0; i < nz; i++)
    {   
        row_elem = I[i];
        col_elem = J[i];
        val_elem = val[i];
#pragma omp critical (ELLPACK_optimization) 
    {

        k = curr_idx_per_row [row_elem];

        if (k >= nz_per_row[row_elem])
        {
            printf("offset maggiore di nz_per_row");
            exit(1);
        }

        if ( k < nz_per_row[row_elem]) {

            if (val != NULL) (*values)[row_elem][k] = val[i];
            (*col_indices)[row_elem][k] = J[i];
            curr_idx_per_row [row_elem] ++;
        }
    }
    }

    free(curr_idx_per_row);
    printf("ELLPACK parallel completed...\n");

    // for (int j = 0; j < rows; j++)
    // {
    //     for(int k = 0; k < max_so_far; k++)
    //     {
    //        AUDIT printf("ELLPACK VALUE: %.66lf - COL: %d\n", (*values)[j][k], (*col_indices)[j][k]);
    //     }
    // }

    return nz_per_row;
}

